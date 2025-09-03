import torch.nn
from torch_geometric.nn import GCNConv
import dgl
import torch.nn as nn
from torch.nn import init
from .base import MLP_spa
from load_data.load_data import load_dataset


train_data = load_dataset("cornell", 0.6, 0.2, 0.2, 0)

def freeze_parameters(parameters):
    for param in parameters:
        if isinstance(param, nn.Parameter):
            param.requires_grad = False

def dgl_to_scipy(graph):
    g_coo = graph.adjacency_matrix()
    return g_coo

def compute_low_freq_matrix(adj):
    return adj


def get_unnormalized_low_freq_matrix(dgl_graph):
    adj = dgl_to_scipy(dgl_graph)
    adj_low_unnormalized = compute_low_freq_matrix(adj)
    return adj_low_unnormalized

def normalize_tensor(mx, eqvar=None):
    rowsum = torch.sum(mx, 1)
    if eqvar:
        r_inv = torch.pow(rowsum, -1 / eqvar).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

    else:
        r_inv = torch.pow(rowsum, -1).flatten()
        r_inv[torch.isinf(r_inv)] = 0.0
        r_mat_inv = torch.diag(r_inv)
        mx = torch.mm(r_mat_inv, mx)
        return mx

def convert_to_dglgraph(data_batch):
    edge_src, edge_dst = data_batch.edge_index
    num_nodes_edges = max(edge_src.max(), edge_dst.max()).item() + 1
    if hasattr(data_batch, 'x'):
        num_nodes_features = data_batch.x.size(0)
    else:
        num_nodes_features = 0
    num_nodes = max(num_nodes_edges, num_nodes_features)
    g = dgl.graph((edge_src, edge_dst), num_nodes=num_nodes)
    if num_nodes_features > 0:
        g.ndata['feature'] = data_batch.x
    if hasattr(data_batch, 'y'):
        g.ndata['label'] = data_batch.y
    if hasattr(data_batch, 'train_mask'):
        g.ndata['train_mask'] = data_batch.train_mask
    if hasattr(data_batch, 'val_mask'):
        g.ndata['val_mask'] = data_batch.val_mask
    if hasattr(data_batch, 'test_mask'):
        g.ndata['test_mask'] = data_batch.test_mask

    return g


def assign_pseudo_labels_to_dglgraph(g, train_nid, labels):
    device = labels.device

    num_nodes = g.number_of_nodes()
    label_unk = (torch.ones(num_nodes, device=device) * -1).long()
    label_unk[train_nid] = labels.long() 

    g.ndata['label_unk'] = label_unk
    return g


def create_block_from_dglgraph(g):
    block = dgl.to_block(g, dst_nodes=torch.arange(g.number_of_nodes()))
    return block

class ACMModule(nn.Module):
    def attention3(self, output_low, output_high, output_mlp):
        T = 3
        output_low, output_high, output_mlp = (
            self.layer_norm_low(output_low),
            self.layer_norm_high(output_high),
            self.layer_norm_mlp(output_mlp),
        )
        low_product = torch.mm(output_low, self.att_vec_low)
        high_product = torch.mm(output_high, self.att_vec_high)
        mlp_product = torch.mm(output_mlp, self.att_vec_mlp)
        a = torch.isnan(low_product).any().item()
        concatenated = torch.cat([low_product, high_product, mlp_product], dim=1)
        sigmoid_result = torch.sigmoid(concatenated)
        mm_result = torch.mm(sigmoid_result, self.att_vec)
        logits = mm_result / T

        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def __init__(self,
                 input_channels: int, hidden_channels: int, output_channels: int, batch_size: int,
                 dropout=0, tail_activation=False, activation=nn.ReLU(inplace=True), gn=False):
        super(ACMModule, self).__init__()
        device = torch.device('cuda')
        self.weight_low, self.weight_high, self.weight_mlp = (
            Parameter(init.xavier_uniform_(torch.FloatTensor(hidden_channels, output_channels).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(hidden_channels, output_channels).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(hidden_channels, output_channels).to(device)))
        )
        self.layer_norm_low, self.layer_norm_high, self.layer_norm_mlp = (
            nn.LayerNorm(output_channels),
            nn.LayerNorm(output_channels),
            nn.LayerNorm(output_channels),
        )
        self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * output_channels, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * output_channels, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * output_channels, 1).to(device))),
        )
        self.att_vec = Parameter(init.xavier_uniform_(torch.FloatTensor(3, 3).to(device)))

    def forward(self, input, adj_low, adj_high, adj_low_unnormalized):
        leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        ab = torch.mm(input, self.weight_low)
        b = ab[2]
        output_low = leaky_relu(torch.spmm(adj_low, ab))
        b = output_low[2]
        output_high = leaky_relu(torch.spmm(adj_high, torch.mm(input, self.weight_high)))
        output_mlp = leaky_relu(torch.mm(input, self.weight_mlp))
        b = output_low[2]

        self.att_low, self.att_high, self.att_mlp = self.attention3(
            (output_low), (output_high), (output_mlp)
        )
        return 3 * (

                self.att_low * output_low
                + self.att_high * output_high
                + self.att_mlp * output_mlp
        )


def mp_func_spa(edges):
    src_label = edges.src['label_unk']
    dst_label = edges.dst['label_unk']

    src_fake_label = torch.zeros_like(src_label)
    src_fake_label[(src_label < 0) | (dst_label < 0)] = 2
    condition = (src_label != dst_label) & (src_label >= 0) & (dst_label >= 0)
    src_fake_label[condition] = 1

    return {
        'm_low': edges.src['h_low'],
        'm_high': edges.src['h_high'],
        'src': edges.src['_ID'],
        'src_fake_label': src_fake_label
    }

def agg_func_spa(nodes):
    return {
        'neigh_hete_low': (nodes.mailbox['m_low'] * (nodes.mailbox['src_fake_label'] == 1).unsqueeze(-1)).sum(1),
        'neigh_hete_high': (nodes.mailbox['m_high'] * (nodes.mailbox['src_fake_label'] == 1).unsqueeze(-1)).sum(1),
        'neigh_homo_low': (nodes.mailbox['m_low'] * (nodes.mailbox['src_fake_label'] == 0).unsqueeze(-1)).sum(1),
        'neigh_homo_high': (nodes.mailbox['m_high'] * (nodes.mailbox['src_fake_label'] == 0).unsqueeze(-1)).sum(1),
        'neigh_unk_low': (nodes.mailbox['m_low'] * (nodes.mailbox['src_fake_label'] == 2).unsqueeze(-1)).sum(1),
        'neigh_unk_high': (nodes.mailbox['m_high'] * (nodes.mailbox['src_fake_label'] == 2).unsqueeze(-1)).sum(1)
    }


def get_spa(nfeat,
        nhid,
        nclass,
        nlayers):
    return ACM(nfeat,
        nhid,
        nclass,
        nlayers,
        dropout=0.1,
        model_type='acm_spa',
        variant=False,
        init_layers_X=1,)


import numpy as np
import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn


device = f"cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)


class GraphConvolution(Module):
    def __init__(
        self,
        in_features,
        out_features,
        model_type,
        output_layer=0,
        variant=False,
    ):
        super(GraphConvolution, self).__init__()
        (
            self.in_features,
            self.out_features,
            self.output_layer,
            self.model_type,
            self.variant,
        ) = (
            in_features,
            out_features,
            output_layer,
            model_type,
            variant,
        )
        self.bn_low = nn.BatchNorm1d(out_features)
        self.bn_high = nn.BatchNorm1d(out_features)
        self.bn_mlp = nn.BatchNorm1d(out_features)
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0
        self.weight_low, self.weight_high, self.weight_mlp = (
            Parameter(init.xavier_uniform_(torch.FloatTensor(in_features, out_features).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(in_features, out_features).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(in_features, out_features).to(device))),
        )
        self.att_vec_low, self.att_vec_high, self.att_vec_mlp = (
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
        )
        self.layer_norm_low, self.layer_norm_high, self.layer_norm_mlp = (
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
            nn.LayerNorm(out_features),
        )
        self.layer_norm_struc_low, self.layer_norm_struc_high = nn.LayerNorm(
            out_features
        ), nn.LayerNorm(out_features)
        self.att_struc_low = Parameter(
            torch.FloatTensor(1 * out_features, 1).to(device)
        )
        self.att_vec_3 = init.xavier_uniform_(Parameter(torch.FloatTensor(3, 3).to(device)))
        self.att_vec_low_fr, self.att_vec_high_fr, self.att_vec_low_be, self.att_vec_high_be, self.att_vec_low_unk, self.att_vec_high_unk = (
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
            Parameter(init.xavier_uniform_(torch.FloatTensor(1 * out_features, 1).to(device))),
        )

        self.att_vec = init.xavier_uniform_(Parameter(torch.FloatTensor(7, 7).to(device)))

        self.fc_balance = MLP_spa(out_features, hidden_channels=128, output_channels=1,
                              num_layers=1)
        self.balance_w = nn.Sigmoid()
        self.fr_low_g = None
        self.fr_high_g = None
        self.mlp_g = None
    def reset_parameters(self):

        stdv = 1.0 / math.sqrt(self.weight_mlp.size(1))
        std_att = 1.0 / math.sqrt(self.att_vec_mlp.size(1))
        std_att_vec = 1.0 / math.sqrt(self.att_vec.size(1))

        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.struc_low.data.uniform_(-stdv, stdv)

        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)
        self.att_struc_low.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

        self.layer_norm_low.reset_parameters()
        self.layer_norm_high.reset_parameters()
        self.layer_norm_mlp.reset_parameters()
        self.layer_norm_struc_low.reset_parameters()
        self.layer_norm_struc_high.reset_parameters()
        std_att_fr = 1.0 / math.sqrt(self.att_vec_low_fr.size(1))
        self.att_vec_low_fr.data.uniform_(-std_att_fr, std_att_fr)
        self.att_vec_high_fr.data.uniform_(-std_att_fr, std_att_fr)
        self.att_vec_low_be.data.uniform_(-std_att_fr, std_att_fr)
        self.att_vec_high_be.data.uniform_(-std_att_fr, std_att_fr)
        self.att_vec_low_unk.data.uniform_(-std_att_fr, std_att_fr)
        self.att_vec_high_unk.data.uniform_(-std_att_fr, std_att_fr)

    def attention3(self, output_low, output_high, output_mlp):
        T = 3
        logits = (
            torch.mm(
                torch.sigmoid(
                    torch.cat(
                        [
                            torch.mm((output_low), self.att_vec_low),
                            torch.mm((output_high), self.att_vec_high),
                            torch.mm((output_mlp), self.att_vec_mlp),
                        ],
                        1,
                    )
                ),
                self.att_vec_3,
            )
            / T
        )
        att = torch.softmax(logits, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def attention7(self, neigh_hete_low, neigh_hete_high, neigh_homo_low, neigh_homo_high, neigh_unk_low, neigh_unk_high,
                   output_mlp, balance_1=None, balance_2=None, balance_3=None):
        T = 7
        if balance_1 != None:
            logits = (
                    torch.mm(
                        torch.sigmoid(
                            torch.cat(
                                [
                                    torch.mm(neigh_hete_low, self.att_vec_low_fr),
                                    torch.mm(neigh_hete_high, self.att_vec_high_fr),
                                    torch.mm(neigh_homo_low, self.att_vec_low_be),
                                    torch.mm(neigh_homo_high, self.att_vec_high_be),
                                    torch.mm(neigh_unk_low, self.att_vec_low_fr) * balance_1 + torch.mm(neigh_unk_low,
                                                                                                        self.att_vec_low_be) * (
                                                1 - balance_1),
                                    torch.mm(neigh_unk_high, self.att_vec_high_fr) * balance_2 + torch.mm(neigh_hete_high,
                                                                                                          self.att_vec_high_be) * (
                                                1 - balance_2),
                                    torch.mm(output_mlp, self.att_vec_mlp),
                                ],
                                1,
                            )
                        ),
                        self.att_vec,
                    )
                    / T
            )
        else:
            logits = (
                    torch.mm(
                        torch.sigmoid(
                            torch.cat(
                                [
                                    torch.mm(neigh_hete_low, self.att_vec_low_fr),
                                    torch.mm(neigh_hete_high, self.att_vec_high_fr),
                                    torch.mm(neigh_homo_low, self.att_vec_low_be),
                                    torch.mm(neigh_homo_high, self.att_vec_high_be),
                                    torch.mm(neigh_unk_low, self.att_vec_low_unk),
                                    torch.mm(neigh_unk_high, self.att_vec_high_unk),
                                    torch.mm(output_mlp, self.att_vec_mlp),
                                ],
                                1,
                            )
                        ),
                        self.att_vec,
                    )
                    / T
            )
        att = torch.softmax(logits, 1)
        return (att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None],
                att[:, 3][:, None], att[:, 4][:, None], att[:, 5][:, None], att[:, 6][:, None])

    def forward(self, input, adj_low, adj_high, adj_low_unnormalized, graph=None, fr_high_g=None, fr_low_g=None, mlp_g=None):


        output_low = F.relu(torch.spmm(adj_low, (torch.mm(input, self.weight_low))))
        output_high = F.relu(torch.spmm(adj_high, (torch.mm(input, self.weight_high))))
        output_mlp = F.relu(torch.mm(input, self.weight_mlp))

        if self.model_type == "acm_spa":
            graph.srcdata["h_low"] = output_low
            graph.srcdata["h_high"] = output_high

            graph.update_all(mp_func_spa, agg_func_spa)

            neigh_hete_low = graph.dstdata["neigh_hete_low"].to(device)
            neigh_hete_high = graph.dstdata["neigh_hete_high"].to(device)
            neigh_homo_low = graph.dstdata["neigh_homo_low"].to(device)
            neigh_homo_high = graph.dstdata["neigh_homo_high"].to(device)
            neigh_unk_low = graph.dstdata["neigh_unk_low"].to(device)
            neigh_unk_high = graph.dstdata["neigh_unk_high"].to(device)

            if fr_high_g != None:
                balance_1 = self.balance_w(self.fc_balance(neigh_unk_low)).to(device)
                balance_2 = self.balance_w(self.fc_balance(neigh_unk_high)).to(device)
                self.att_fr_low, self.att_fr_high, self.att_be_low, self.att_be_high, self.att_unk_low, self.att_unk_high, self.att_mlp = self.attention7(
                    neigh_hete_low, neigh_hete_high, neigh_homo_low, neigh_homo_high, neigh_unk_low, neigh_unk_high, output_mlp, balance_1, balance_2
                )



                return 7 * (
                        self.att_fr_low * neigh_hete_low +
                        self.att_fr_high * neigh_hete_high +
                        self.att_be_low * neigh_homo_low +
                        self.att_be_high * neigh_homo_high +
                        self.att_unk_low * neigh_unk_low +
                        self.att_unk_high * neigh_unk_high +
                        self.att_mlp * output_mlp
                )

            self.att_fr_low, self.att_fr_high, self.att_be_low, self.att_be_high, self.att_unk_low, self.att_unk_high, self.att_mlp = self.attention7(
                neigh_hete_low, neigh_hete_high, neigh_homo_low, neigh_homo_high, neigh_unk_low, neigh_unk_high, output_mlp
            )

            return 7 * (
                    self.att_fr_low * neigh_hete_low +
                    self.att_fr_high * neigh_hete_high +
                    self.att_be_low * neigh_homo_low +
                    self.att_be_high * neigh_homo_high +
                    self.att_unk_low * neigh_unk_low +
                    self.att_unk_high * neigh_unk_high +
                    self.att_mlp * output_mlp
            )
        if self.model_type == "acmgcn" or self.model_type == "acmsnowball":
            self.att_low, self.att_high, self.att_mlp = self.attention3(
                (output_low), (output_high), (output_mlp)
            )
            return 3 * (
                self.att_low * output_low
                + self.att_high * output_high
                + self.att_mlp * output_mlp
            )
        else:

            self.att_low, self.att_high, self.att_mlp = self.attention3(
                (output_low), (output_high), (output_mlp)
            )
            return 3 * (
                self.att_low * output_low
                + self.att_high * output_high
                + self.att_mlp * output_mlp
            )


device = f"cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

class ACM(nn.Module):
    def __init__(
        self,
        nfeat,
        nhid,
        nclass,
        nlayers,
        dropout,
        model_type,
        variant=False,
        init_layers_X=1,
    ):
        super(ACM, self).__init__()
        self.bn = nn.BatchNorm1d(nhid)
        self.preprocessed = False
        self.gcns, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.model_type, self.nlayers = (model_type, nlayers)
        if (
            self.model_type == "acmgcn"
            or self.model_type == "acmgcnp"
            or self.model_type == "acmgcnpp"
            or self.model_type == "acm_spa"
        ):
            self.gcns.append(
                GraphConvolution(
                    nfeat,
                    nhid,
                    model_type=model_type,
                    variant=variant,
                )
            )
            self.gcns.append(
                GraphConvolution(
                    1 * nhid,
                    nclass,
                    model_type=model_type,
                    output_layer=1,
                    variant=variant
                )
            )

        self.dropout = dropout
        self.fea_param, self.xX_param = init.xavier_uniform_(Parameter(
            torch.FloatTensor(1, 1).to(device))
        ), init.xavier_uniform_(Parameter(torch.FloatTensor(1, 1).to(device)))

        self.reset_parameters()

    def reset_parameters(self):
        if self.model_type == "acmgcnpp":
            self.mlpX.reset_parameters()
        else:
            pass

    def preprocess(self, data):
        device = torch.device('cuda')
        g = convert_to_dglgraph(data)
        train_nid = torch.nonzero(data.train_mask, as_tuple=False).squeeze()
        labels = data.y[train_nid]
        g = assign_pseudo_labels_to_dglgraph(g, train_nid, labels)
        self.block = create_block_from_dglgraph(g)
        self.h = data.x


    def forward(self, data, adj_low_un, adj_low, adj_high, fr_low_g=None, fr_high_g=None, mlp_g=None):
        if fr_low_g !=None:
            freeze_parameters(fr_low_g)
            freeze_parameters(fr_low_g)
            mlp_g
        if self.preprocessed:
            block = self.block
            h = self.h
        else:
            self.preprocess(data)
            block = self.block
        x = data.x
        if (
            self.model_type == "acmgcn"
            or self.model_type == "acmsgc"
            or self.model_type == "acmsnowball"
            or self.model_type == "acmgcnp"
            or self.model_type == "acmgcnpp"
            or self.model_type == "acm_spa"
        ):

            x = F.dropout(x, self.dropout, training=self.training)
            if self.model_type == "acmgcnpp":
                xX = F.dropout(
                    F.relu(self.mlpX(x, input_tensor=True)),
                    self.dropout,
                    training=self.training,
                )
        if self.model_type == "acmsnowball":
            list_output_blocks = []
            for layer, layer_num in zip(self.gcns, np.arange(self.nlayers)):
                if layer_num == 0:
                    list_output_blocks.append(
                        F.dropout(
                            F.relu(layer(x, adj_low, adj_high)),
                            self.dropout,
                            training=self.training,
                        )
                    )
                else:
                    list_output_blocks.append(
                        F.dropout(
                            F.relu(
                                layer(
                                    torch.cat([x] + list_output_blocks[0:layer_num], 1),
                                    adj_low,
                                    adj_high,
                                )
                            ),
                            self.dropout,
                            training=self.training,
                        )
                    )
            return self.gcns[-1](
                torch.cat([x] + list_output_blocks, 1), adj_low, adj_high
            )

        if (
                self.model_type == "acmgcn"
                or self.model_type == "acmsgc"
                or self.model_type == "acmsnowball"
                or self.model_type == "acmgcnp"
                or self.model_type == "acmgcnpp"
        ):
            fea1 = self.gcns[0](x, adj_low, adj_high, adj_low_un)
        elif (self.model_type == "acm_spa"):
            if fr_high_g != None:
                fea1 = self.gcns[0](x, adj_low, adj_high, adj_low_un, block, fr_high_g[0], fr_low_g[0], mlp_g[0])
            else:
                fea1 = self.gcns[0](x, adj_low, adj_high, adj_low_un, block)

        if (
            self.model_type == "acmgcn"
            or self.model_type == "acmgcnp"
            or self.model_type == "acmgcnpp"
        ):

            fea1 = F.dropout((F.relu(fea1)), self.dropout, training=self.training)
            fea2 = self.gcns[1](fea1, adj_low, adj_high, adj_low_un)

        elif self.model_type == "acm_spa":
            fea1 = F.relu(fea1)
            fea1 = F.dropout(fea1, self.dropout, training=self.training)
            if fr_high_g != None:
                fea2 = self.gcns[1](fea1, adj_low, adj_high, adj_low_un, block, fr_high_g[1], fr_low_g[1], mlp_g[1])
            else:
                fea2 = self.gcns[1](fea1, adj_low, adj_high, adj_low_un, block)
        return F.log_softmax(fea2, dim=1)










