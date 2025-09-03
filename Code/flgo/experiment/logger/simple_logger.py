from flgo.experiment.logger import BasicLogger
import numpy as np
import flgo.simulator.base as ss
import torch
from torch_geometric.utils import to_scipy_sparse_matrix
import numpy as np
from scipy.linalg import eigh
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.neighbors import KernelDensity
from torch_geometric.utils import to_dense_adj
import dgl

def dgl_to_scipy(graph):
    """
    将DGL图转换为Scipy稀疏矩阵形式。
    """
    # DGL图转换为COO格式（坐标列表），然后转换为Scipy稀疏矩阵
    g_coo = graph.adjacency_matrix()
    # 返回Scipy稀疏矩阵
    return g_coo
def compute_low_freq_matrix(adj):
    """
    计算给定邻接矩阵的低频版本。
    此处简单返回未归一化的邻接矩阵作为示例。
    在实际应用中，你可能需要根据低频处理的定义对邻接矩阵进行更复杂的变换。
    """
    # 示例中我们简单返回原始矩阵。根据需要进行变换。
    return adj
def get_unnormalized_low_freq_matrix(dgl_graph):
    """
    从DGL图获取未归一化的低频邻接矩阵。
    """
    # 将DGL图转换为Scipy稀疏矩阵
    adj = dgl_to_scipy(dgl_graph)
    # 计算低频矩阵
    adj_low_unnormalized = compute_low_freq_matrix(adj)
    return adj_low_unnormalized
def normalize_tensor(mx, eqvar=None):
    """
    Row-normalize sparse matrix
    """
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


def getsc(A1,X1,A2,X2):


    # Calculate normalized Laplacians
    L1 = calculate_normalized_laplacian(A1)
    L2 = calculate_normalized_laplacian(A2)

    # Calculate eigenvalues and eigenvectors
    eigvals1, eigvecs1 = eigh(L1)
    eigvals2, eigvecs2 = eigh(L2)

    # Calculate LED for each graph
    led1 = calculate_led(eigvecs1, X1)
    led2 = calculate_led(eigvecs2, X2)

    # Calculate KDEs for the LEDs
    kde1 = calculate_kde(led1)
    kde2 = calculate_kde(led2)

    # Define the grid for SC calculation
    grid = np.linspace(0, max(led1.max(), led2.max()), 1000)

    # Calculate SC
    sc = calculate_sc(kde1, kde2, grid)

    return sc

def convert_to_dglgraph(data_batch):
    # 根据edge_index构建DGLGraph
    edge_src, edge_dst = data_batch.edge_index
    num_nodes_edges = max(edge_src.max(), edge_dst.max()).item() + 1  # 边索引中的最大节点ID + 1

    # 如果data_batch具有节点特征，用它们来确定节点数量
    if hasattr(data_batch, 'x'):
        num_nodes_features = data_batch.x.size(0)
    else:
        num_nodes_features = 0

    # 确定图中的节点总数
    num_nodes = max(num_nodes_edges, num_nodes_features)
    device = data_batch.x.device
    # 创建图
    g = dgl.graph((edge_src, edge_dst), num_nodes=num_nodes).to(device)


    # 如果有节点特征，设置节点特征
    if hasattr(data_batch, 'x'):
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


def get_adjacency_matrix(data):
    # 获取节点的数量
    num_nodes = data.num_nodes if hasattr(data, 'num_nodes') else data.x.size(0)

    # 使用to_dense_adj函数将edge_index转换为邻接矩阵，并确保邻接矩阵包含所有节点
    adj_matrix = to_dense_adj(edge_index=data.edge_index, max_num_nodes=num_nodes)

    # adj_matrix的形状是[1, N, N]，我们只需要[N, N]的矩阵，所以使用squeeze方法去除第一维
    adj_matrix = adj_matrix.squeeze(0)

    return adj_matrix


def calculate_normalized_laplacian(A):
    D = np.diag(np.sum(A, axis=1))
    # 使用伪逆代替标准逆，以处理奇异矩阵问题
    D_inv_sqrt = np.linalg.pinv(np.sqrt(D))
    L_norm = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt
    return L_norm


def calculate_led(eigenvectors, X):
    X_hat = eigenvectors.T @ X  # Transform features to the spectral domain
    led = np.sum(X_hat ** 2, axis=1)
    led /= np.sum(led)
    return led


def calculate_kde(led, bandwidth='silverman'):
    kde = gaussian_kde(led, bw_method=bandwidth)
    return kde


def calculate_sc(kde1, kde2, grid):
    p1 = kde1(grid)
    p2 = kde2(grid)
    sc = jensenshannon(p1, p2) ** 2
    return sc


def kde_prob_estimation(features, bandwidth=0.1):
    # Convert to numpy if the input is a Tensor
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()

    # Initialize the KernelDensity object
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')

    # Fit the KDE model, this is where we calculate the KDE for our data
    kde.fit(features)

    # Compute the log probability density for each feature vector
    log_prob_densities = kde.score_samples(features)

    # Convert log probabilities to probabilities
    prob_densities = np.exp(log_prob_densities)

    return prob_densities


def compute_js_divergence(data1, data2, bandwidth=0.3975):
    # KDE probability estimation for both graphs
    prob_distr_1 = kde_prob_estimation(data1.data.x, bandwidth)
    prob_distr_2 = kde_prob_estimation(data2.data.x, bandwidth)

    # Ensure that probability distributions sum to 1
    prob_distr_1 /= prob_distr_1.sum()
    prob_distr_2 /= prob_distr_2.sum()

    # Compute Jensen-Shannon Divergence
    js_divergence = jensenshannon(prob_distr_1, prob_distr_2, base=2)

    return js_divergence

def calculate_high_frequency_feature(data):
    x = data.train_mask.to(torch.float32)
    edge_index = torch.tensor(data.edge_index, dtype=torch.int64)

    adj = to_scipy_sparse_matrix(edge_index, num_nodes=x.size(0))

    degrees = adj.sum(axis=1).A.flatten()
    D = torch.diag(torch.tensor(degrees))

    L = D - torch.tensor(adj.toarray(), dtype=torch.float32)

    xTLx = torch.matmul(x.t(), torch.matmul(L, x))

    # 计算 x^T x
    xTx = torch.matmul(x.t(), x)

    # 计算 S_high
    S_high = xTLx / xTx


    return S_high.item()  # 返回计算结果的数值
class SimpleLogger(BasicLogger):
    r"""Simple Logger. Only evaluating model performance on testing dataset and validation dataset."""
    def initialize(self):
        """This method is used to record the stastic variables that won't change across rounds (e.g. local_movielens_recommendation data size)"""
        for c in self.participants:
            self.output['client_datavol'].append(len(c.train_data))

    """This logger only records metrics on validation dataset"""
    def log_once(self, *args, **kwargs):
        if self.current_round == -1:
            device = torch.device('cuda')
            for client in self.clients:
                g = convert_to_dglgraph(client.train_data.data)
                adj_low_unnormalized = get_unnormalized_low_freq_matrix(g).to(device)
                eye_matrix = torch.eye(g.number_of_nodes()).to(device)
                adj_low = (normalize_tensor(eye_matrix + adj_low_unnormalized.to_dense())).to(device)
                adj_high = (torch.eye(g.number_of_nodes()).to(device) - adj_low).to(device).to_sparse()
                client.adj_low_un = adj_low_unnormalized
                client.adj_low = adj_low
                client.adj_high = adj_high

        self.info('Current_time:{}'.format(self.clock.current_time))
        self.output['time'].append(self.clock.current_time)
        test_metric = self.coordinator.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        val_metrics = self.coordinator.global_test(flag='val')
        if self.current_round == 5:
            '''A1 = np.array(get_adjacency_matrix(self.clients[0].train_data.data))
            X1 = np.array(self.clients[0].train_data.data.x)

            spectral_feature = []'''

            '''for client in self.clients:
                A2 = np.array(get_adjacency_matrix(client.train_data.data))
                X2 = np.array(client.train_data.data.x)
                if A1.shape != A2.shape:
                    # 如果维度不同，跳过当前client
                    continue

                spectral_feature.append(getsc(A1, X1, A2, X2))'''

            '''import matplotlib.pyplot as plt

            # 创建颜色列表，每个点一个颜色
            colors = plt.cm.rainbow(np.linspace(0, 1, 10))

            # 绘图
            plt.figure(figsize=(10, 6))
            # 确定spectral_feature列表的长度
            length_of_features = len(spectral_feature)

            # 为spectral_feature中的每个点进行散点图绘制
            for i in range(length_of_features):
                plt.scatter(spectral_feature[i], val_metrics['accuracy'][i], color=colors[i % len(colors)], s=100,
                            label=f'point {i + 1}')

            # 添加图例
            plt.legend()

            # 设置x轴和y轴的标签
            plt.xlabel('Spectral Feature')
            plt.ylabel('Accuracy')
            plt.xlim(0.0, 0.5)
            plt.ylim(0.0, 0.05)

            # 显示图表
            plt.show()'''

        local_data_vols = [c.datavol for c in self.participants]
        total_data_vol = sum(local_data_vols)
        for met_name, met_val in val_metrics.items():
            self.output['val_'+met_name+'_dist'].append(met_val)
            self.output['val_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            self.output['mean_val_' + met_name].append(np.mean(met_val))
            self.output['std_val_' + met_name].append(np.std(met_val))
        # output to stdout
        self.show_current_output()