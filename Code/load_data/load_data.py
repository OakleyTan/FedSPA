from torch_geometric.datasets import Planetoid, Coauthor, WebKB, HeterophilousGraphDataset
import torch
import numpy as np
from torch_geometric.data import Data
from ogb.nodeproppred import NodePropPredDataset
import scipy.io

def rand_train_test_idx(label, train_prop, valid_prop, test_prop):
    labeled_nodes = torch.where(label != -1)[0]
    n = labeled_nodes.shape[0]
    train_num = int(n * train_prop)
    valid_num = int(n * valid_prop)
    test_num = int(n * test_prop)

    perm = torch.as_tensor(np.random.permutation(n))
    train_indices = perm[:train_num]
    val_indices = perm[train_num:train_num + valid_num]
    test_indices = perm[train_num + valid_num:train_num + valid_num + test_num]

    return {'train': train_indices.numpy(), 'valid': val_indices.numpy(), 'test': test_indices.numpy()}

def index_to_mask(splits_lst, num_nodes):
    mask_len = len(splits_lst)
    train_mask = torch.zeros((mask_len, num_nodes), dtype=torch.bool)
    val_mask = torch.zeros((mask_len, num_nodes), dtype=torch.bool)
    test_mask = torch.zeros((mask_len, num_nodes), dtype=torch.bool)

    for i in range(mask_len):
        train_mask[i][splits_lst[i]['train']] = True
        val_mask[i][splits_lst[i]['valid']] = True
        test_mask[i][splits_lst[i]['test']] = True

    return train_mask.T, val_mask.T, test_mask.T

def load_dataset(dataname, train_prop, valid_prop, test_prop, num_masks):
    assert dataname in ('cora', 'cs', 'physics', 'cornell', 'wisconsin', 'minesweeper', 'arxiv-year'), 'Invalid dataset'

    if dataname in ['cora']:
        dataset = Planetoid(root='/tmp/Cora', name=dataname)
        data = dataset[0]
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname in ['cs', 'physics']:
        dataset = Coauthor(root='/tmp/Coauthor', name=dataname)
        data = dataset[0]
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname in ['cornell', 'wisconsin']:
        dataset = WebKB(root='/tmp/WebKB', name=dataname)
        data = dataset[0]
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    elif dataname == 'minesweeper':
        dataset = HeterophilousGraphDataset(root='dsd', name=dataname)
        data = dataset[0]
        # data.y = data.y.squeeze()
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)


    elif dataname == 'arxiv-year':
        ogb_dataset = NodePropPredDataset(name='ogbn-arxiv')
        graph = ogb_dataset.graph
        edge_index = torch.as_tensor(graph['edge_index'])
        x = torch.as_tensor(graph['node_feat'])
        label = even_quantile_labels(graph['node_year'].flatten(), 5, verbose=False)
        y = torch.as_tensor(label).reshape(-1)
        data = Data(x=x, edge_index=edge_index, y=y)
        splits_lst = [rand_train_test_idx(data.y, train_prop=train_prop, valid_prop=valid_prop, test_prop=test_prop)
                      for _ in range(num_masks)]
        data.train_mask, data.val_mask, data.test_mask = index_to_mask(splits_lst, data.num_nodes)

    return data

def even_quantile_labels(vals, nclasses, verbose=True):
    label = -1 * np.ones(vals.shape[0], dtype=np.int)
    interval_lst = []
    lower = -np.inf
    for k in range(nclasses - 1):
        upper = np.nanquantile(vals, (k + 1) / nclasses)
        interval_lst.append((lower, upper))
        inds = (vals >= lower) * (vals < upper)
        label[inds] = k
        lower = upper
    label[vals >= lower] = nclasses - 1
    interval_lst.append((lower, np.inf))
    if verbose:
        print('Class Label Intervals:')
        for class_idx, interval in enumerate(interval_lst):
            print(f'Class {class_idx}: [{interval[0]}, {interval[1]})]')
    return label

