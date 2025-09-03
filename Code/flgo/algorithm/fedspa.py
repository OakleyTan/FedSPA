import copy
from .fedbase import BasicServer
from .fedbase import BasicParty
from flgo.utils import fmodule
import collections
import flgo.benchmark
import math
import numpy as np

import torch
import torch.nn.functional as F
def sum_of_layer_fisher_differences(fish_dict1, fish_dict2, layer):
    total_difference = 0.0
    for key in fish_dict1:
        if layer in key:
            difference = fish_dict1[key] - fish_dict2[key]
            total_difference += torch.sum(torch.abs(difference))
    return total_difference

def normalize_list(values):
    total = sum(values)
    if total == 0:
        raise ValueError("Sum of the elements is zero, cannot normalize.")
    normalized_values = [value / total for value in values]
    return normalized_values


def calculate_fisher_average(fisher_dicts):
    if not fisher_dicts:
        raise ValueError("The list of Fisher dictionaries is empty.")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    fisher_avg = {}
    for key in fisher_dicts[0].keys():
        fisher_avg[key] = torch.zeros_like(fisher_dicts[0][key], device=device)

    for fish_dict in fisher_dicts:
        for key in fish_dict:
            fisher_avg[key] += fish_dict[key].to(device)
    num_dicts = len(fisher_dicts)
    for key in fisher_avg:
        fisher_avg[key] /= num_dicts

    return fisher_avg


def normalize(data):
    min_val = min(data)
    max_val = max(data)
    range = max_val - min_val
    normalized_data = [(x - min_val) / range for x in data]
    return normalized_data


def calculate_weights(differences):
    float_diff = [x.item() if isinstance(x, torch.Tensor) else x for x in differences]
    normalized_diff = normalize(float_diff)
    pre_weights = [np.exp(-x) for x in normalized_diff]
    sum_weights = sum(pre_weights)
    weights = [w / sum_weights for w in pre_weights]
    return weights

def fish_cal(net, data, adj_low_un, adj_low, adj_high, high_list, low_list, mlp_list):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    label = data.y
    data, label = data.to(device), label.to(device)
    output = net(data, adj_low_un, adj_low, adj_high, high_list, low_list, mlp_list)
    pre = F.log_softmax(output, dim=1)
    log_likelihoods = pre.gather(1, label.view(-1, 1)).squeeze()
    net.zero_grad()
    log_likelihoods.mean().backward()
    average_grads = {name: param.grad.clone() for name, param in net.named_parameters() if param.grad is not None}
    fish_dict = {name: grad ** 2 for name, grad in average_grads.items()}
    return fish_dict


class Server(BasicServer):
    def check_nan_layers(self, models):
        for idx, model in enumerate(models):
            for name, param in model.named_parameters():
                if torch.isnan(param).any():
                    self.gv['logger']['info'](f"Model {idx} layer '{name}' contains NaN values.")


    def aggregate(self, models: list, *args, **kwargs):
        if len(models) == 0: return self.model

        nan_exists = [m.has_nan() for m in models]
        if any(nan_exists):
            if all(nan_exists): raise ValueError("All the received local models have parameters of nan value.")

            self.gv.logger.info(
                'Warning("There exists nan-value in local models, which will be automatically removed from the aggregation list.")')

            new_models = []
            received_clients = []

            for ni, mi, cid in zip(nan_exists, models, self.received_clients):
                if ni: continue
                new_models.append(mi)
                received_clients.append(cid)
            self.received_clients = received_clients
            models = new_models

        local_data_vols = [c.datavol for c in self.clients]
        total_data_vol = sum(local_data_vols)

        fishs = []
        for model, cid in zip(models, self.received_clients):
            fishs.append(fish_cal(model, self.clients[cid].train_data.data, self.clients[cid].adj_low_un,
                                  self.clients[cid].adj_low, self.clients[cid].adj_high,
                                  self.clients[cid].fr_high_g_list, self.clients[cid].fr_low_g_list,
                                  self.clients[cid].mlp_list))

        fish_g = calculate_fisher_average(fishs)

        aggregated_params = collections.defaultdict(lambda: 0)
        if not hasattr(self, 'diffs'):
            self.diffs = []
            for client in self.clients:
                self.diffs.append(abs(client.radio_diff))
            self.weights2 = calculate_weights(self.diffs)
        fisher_weights = {}
        for layer in ["fr", "be"]:
            layer_fish_diffs = []
            for fish in fishs:
                layer_fish_diffs.append(sum_of_layer_fisher_differences(fish, fish_g, layer))
            layer_fish_diffs = normalize_list(layer_fish_diffs)
            fisher_weights[layer] = calculate_weights(layer_fish_diffs)

        for i, model in enumerate(models):
            for name, param in model.named_parameters():
                if any(layer in name for layer in fisher_weights.keys()):
                    for layer in fisher_weights.keys():
                        if layer in name:
                            aggregated_params[name] += param.data * fisher_weights[layer][i]
                else:
                    aggregated_params[name] += param.data * self.weights2[i]
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_params:
                    param.data.copy_(aggregated_params[name])

        return self.model


class Client(BasicParty):
    TaskCalculator = flgo.benchmark.base.BasicTaskCalculator
    def __init__(self, option={}):
        super().__init__()
        self.id = None
        self.data_loader = None
        self.test_data = None
        self.val_data = None
        self.train_data = None
        self.model = None
        self.device = self.gv.apply_for_device()
        self.calculator = self.TaskCalculator(self.device, option['optimizer'])
        self._train_loader = None
        self.optimizer_name = option['optimizer']
        self.learning_rate = option['learning_rate']
        self.momentum = option['momentum']
        self.weight_decay = option['weight_decay']
        self.batch_size = option['batch_size']
        self.num_steps = option['num_steps']
        self.num_epochs = option['num_epochs']
        self.clip_grad = option['clip_grad']
        self.model = None
        self.test_batch_size = option['test_batch_size']
        self.loader_num_workers = option['num_workers']
        self.current_steps = 0
        self._effective_num_steps = self.num_steps
        self._latency = 0
        self.server = None
        self.option = option
        self.actions = {0: self.reply}
        self.default_action = self.reply

        self.model_g = None

        self.fr_low_g_list = None
        self.fr_high_g_list = None
        self.mlp_list = None

    @fmodule.with_multi_gpus
    def train(self, model):
        r"""
        Standard local_movielens_recommendation training procedure. Train the transmitted model with
        local_movielens_recommendation training dataset.

        Args:
            model (FModule): the global model
        """
        model.train()

        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, weight_decay=self.weight_decay,
                                                  momentum=self.momentum)
        if self.model_g is not None:
            for param in self.model_g.parameters():
                param.requires_grad = False

        self.fr_low_g_list = []
        self.fr_high_g_list = []
        self.mlp_list = []

        for iter in range(self.num_steps):
            if iter == 0:
                for gcn_layer in model.model.gcns:
                    fr_low_g = gcn_layer.att_vec_low_unk
                    self.fr_low_g_list.append(fr_low_g)

                    fr_high_g = gcn_layer.att_vec_high_unk
                    self.fr_high_g_list.append(fr_high_g)

                    mlp_g = gcn_layer.att_vec_mlp
                    self.mlp_list.append(mlp_g)
            batch_data = self.get_batch_data()
            model.zero_grad()
            loss = self.calculator.compute_loss(model, batch_data, self.adj_low_un, self.adj_low, self.adj_high, self.fr_low_g_list, self.fr_high_g_list, self.mlp_list)['loss']
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=1.0)
            optimizer.step()
        return

    @fmodule.with_multi_gpus
    def test(self, model, flag='val'):
        dataset = getattr(self, flag + '_data') if hasattr(self, flag + '_data') else None
        if dataset is None: return {}
        if self.fr_high_g_list != None:
            return self.calculator.test(model, dataset, self.adj_low_un, self.adj_low, self.adj_high, min(self.test_batch_size, len(dataset)), self.option['num_workers'], self.fr_high_g_list, self.fr_low_g_list, self.mlp_list)
        else:
            return self.calculator.test(model, dataset, self.adj_low_un, self.adj_low, self.adj_high, min(self.test_batch_size, len(dataset)), self.option['num_workers'])

    def unpack(self, received_pkg):
        # unpack the received package
        return received_pkg['model']

    def reply(self, svr_pkg):
        model = self.unpack(svr_pkg)
        self.model_g = copy.deepcopy(model)
        self.train(model)
        cpkg = self.pack(model)
        return cpkg

    def pack(self, model, *args, **kwargs):
        return {
            "model": model,
            "model_g": model
        }

    def is_idle(self):
        r"""
        Check if the client is available to participate training.

        Returns:
            True if the client is available according to the active_rate else False
        """
        return self.gv.simulator.client_states[self.id] == 'idle'

    def is_dropped(self):
        r"""
        Check if the client drops out during communicating.

        Returns:
            True if the client was being dropped
        """
        return self.gv.simulator.client_states[self.id] == 'dropped'

    def is_working(self):
        r"""
        Check if the client is training the model.

        Returns:
            True if the client is working
        """

        return self.gv.simulator.client_states[self.id] == 'working'

    def train_loss(self, model):
        r"""
        Get the loss value of the model on local_movielens_recommendation training data

        Args:
            model (flgo.utils.fmodule.FModule|torch.nn.Module): model

        Returns:
            the training loss of model on self's training data
        """
        return self.test(model, 'train')['loss']

    def val_loss(self, model):
        r"""
        Get the loss value of the model on local_movielens_recommendation validating data

        Args:
            model (flgo.utils.fmodule.FModule|torch.nn.Module): model

        Returns:
            the validation loss of model on self's validation data
        """
        return self.test(model)['loss']

    def register_server(self, server=None):
        r"""
        Register the server to self.server
        """
        self.register_objects([server], 'server_list')
        if server is not None:
            self.server = server

    def set_local_epochs(self, epochs=None):
        r"""
        Set local_movielens_recommendation training epochs
        """
        if epochs is None: return
        self.epochs = epochs
        self.num_steps = self.epochs * math.ceil(len(self.train_data) / self.batch_size)
        return

    def set_batch_size(self, batch_size=None):
        r"""
        Set local_movielens_recommendation training batch size

        Args:
            batch_size (int): the training batch size
        """
        if batch_size is None: return
        self.batch_size = batch_size

    def set_learning_rate(self, lr=None):
        """
        Set the learning rate of local_movielens_recommendation training
        Args:
            lr (float): a real number
        """
        self.learning_rate = lr if lr else self.learning_rate

    def get_time_response(self):
        """
        Get the latency amount of the client

        Returns:
            self.latency_amount if client not dropping out
        """
        return np.inf if self.dropped else self.time_response

    def get_batch_data(self):
        """
        Get the batch of training data
        Returns:
            a batch of data
        """
        if self._train_loader is None:
            self._train_loader = self.calculator.get_dataloader(self.train_data, batch_size=self.batch_size,
                                                                   num_workers=self.loader_num_workers,
                                                                   pin_memory=self.option['pin_memory'], drop_last=not self.option.get('no_drop_last', False))
        try:
            batch_data = next(self.data_loader)
        except Exception as e:
            self.data_loader = iter(self._train_loader)
            batch_data = next(self.data_loader)
        self.current_steps = (self.current_steps + 1) % self.num_steps
        if self.current_steps == 0:
            self.data_loader = None
            self._train_loader = None
        return batch_data

    def update_device(self, dev):
        self.device = dev
        self.calculator = self.gv.TaskCalculator(dev, self.calculator.optimizer_name)
