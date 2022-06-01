import numpy as np
import random
import torch
import torch.nn as nn
import dgl
from dgl.data.utils import load_graphs
from ogb.nodeproppred import DglNodePropPredDataset
import os
import torch.nn.functional as F
import json
from ogb.nodeproppred import Evaluator


# convert the inputs from cpu to gpu, accelerate the running speed
def convert_to_gpu(*data, device: str):
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]
    return res


def set_random_seed(seed: int = 0):
    """
    set random seed.
    :param seed: int, random seed to use
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    dgl.random.seed(0)


def load_model(model: nn.Module, model_path: str):
    """Load the model.
    :param model: model
    :param model_path: model path
    """
    print(f"load model {model_path}")
    model.load_state_dict(torch.load(model_path))


def get_n_params(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: model
    :return:
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def load_dataset(root_path: str, dataset_name: str):
    """
    load dataset
    :param root_path: root path
    :param dataset_name: dataset name
    :return:
    """
    if dataset_name == 'ogbn-arxiv':
        dataset = DglNodePropPredDataset(name=dataset_name, root=root_path)

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, labels = dataset[0]  # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)

        # convert graph to be undirected
        src, dst = graph.edges()
        graph.add_edges(dst, src)

        # add self loop
        graph = graph.remove_self_loop().add_self_loop()

        labels = labels.squeeze(dim=-1)
        num_classes = len(labels.unique())
    
    elif dataset_name == 'ogbn-mag':
        graph_list, labels = load_graphs(os.path.join(root_path, 'OGB_MAG/OGB_MAG.pkl'))

        graph = graph_list[0]

        split_idx = torch.load(os.path.join(root_path, 'OGB_MAG/OGB_MAG_split_idx.pkl'))
        train_idx, valid_idx, test_idx = split_idx['train']['paper'], split_idx['valid']['paper'], split_idx['test']['paper']

        labels = labels['paper'].squeeze(dim=-1)
        num_classes = len(labels.unique())

        # remove the structural feature of paper, only keep the original feature
        graph.nodes['paper'].data['feat'] = graph.nodes['paper'].data['feat'][:, :-graph.nodes['author'].data['feat'].shape[1]]

        # Convert to homogeneous graph
        target_type_id = graph.get_ntype_id("paper")
        graph = dgl.to_homogeneous(graph, ndata=["feat"])
        # get the start index of target node type
        start_idx = (graph.ndata[dgl.NTYPE] == target_type_id).nonzero().squeeze(dim=-1)[0].item()
        end_idx = (graph.ndata[dgl.NTYPE] == target_type_id).nonzero().squeeze(dim=-1)[-1].item()
        # get the real index in the homogeneous graph
        train_idx, valid_idx, test_idx = train_idx + start_idx, valid_idx + start_idx, test_idx + start_idx
        # padding labels
        labels = torch.cat([-torch.ones(start_idx).long(), labels], dim=0)
        labels = torch.cat([labels, -torch.ones(graph.number_of_nodes() - end_idx - 1).long()], dim=0)

    else:
        raise ValueError(f'Wrong dataset name: {dataset_name}')

    return graph, labels, num_classes, train_idx, valid_idx, test_idx


def get_node_data_loader(node_neighbors_min_num: int, n_layers: int,
                         graph: dgl.DGLGraph, batch_size: int, train_idx: torch.Tensor, valid_idx: torch.Tensor, test_idx: torch.Tensor,
                         sampled_node_type: str = None, full_neighbors: bool = False,
                         shuffle: bool = True, drop_last: bool = False, num_workers: int = 4):
    """
    get graph node data loader, including train_loader, val_loader and test_loader
    :return:
    """
    if full_neighbors:
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(n_layers)
    else:
        # list of neighbors to sample per edge type for each GNN layer
        sample_nodes_num = []
        for layer in range(n_layers):
            sample_nodes_num.append({etype: node_neighbors_min_num - 5 * layer if node_neighbors_min_num - 5 * layer > 0 else 5 for etype in graph.canonical_etypes})

        # neighbor sampler
        sampler = dgl.dataloading.MultiLayerNeighborSampler(sample_nodes_num)

    if sampled_node_type is None:
        train_loader = dgl.dataloading.NodeDataLoader(
            graph, train_idx, sampler, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

        val_loader = dgl.dataloading.NodeDataLoader(
            graph, valid_idx, sampler, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

        test_loader = dgl.dataloading.NodeDataLoader(
            graph, test_idx, sampler, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    else:
        train_loader = dgl.dataloading.NodeDataLoader(
            graph, {sampled_node_type: train_idx}, sampler, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

        val_loader = dgl.dataloading.NodeDataLoader(
            graph, {sampled_node_type: valid_idx}, sampler, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

        test_loader = dgl.dataloading.NodeDataLoader(
            graph, {sampled_node_type: test_idx}, sampler, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return train_loader, val_loader, test_loader


def get_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, weight_deacy: float):
    """
    get optimizer
    :param model:
    :param optimizer_name:
    :param learning_rate:
    :param weight_deacy:
    :return:
    """
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_deacy)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_deacy)
    else:
        raise ValueError(f"wrong value for optimizer {optimizer_name}!")

    return optimizer


def get_lr_scheduler(optimizer: torch.optim.lr_scheduler, learning_rate: float, t_max: int):
    """
    get lr scheduler
    :param learning_rate:
    :param max_epoch:
    :return:
    """
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=learning_rate / 100)

    return scheduler


def get_loss_func(dataset_name: str, reduction: str = 'mean'):
    if dataset_name in ['ogbn-arxiv', 'ogbn-mag']:
        loss_func = nn.CrossEntropyLoss(reduction=reduction)
    else:
        raise ValueError(f'Unknown loss function for dataset {dataset_name}')
    return loss_func


def generate_hetero_graph(graph: dgl.DGLGraph, num_classes: int):
    """
    generate heterogeneous graph including original nodes and label nodes
    :param graph: dgl.DGLGraph, input original graph
    :param num_classes: int, number of node labels
    :return: the generated heterogeneous graph
    """

    # add original nodes and label nodes
    data_dict = {}
    assert len(graph.canonical_etypes) == 1
    # the generated heterogeneous graph contains three types of relations:
    # 1) node-node relation -> keep the same with original graph; 2) rev node-node relation;
    # 3) node-label relation; 4) label-node relation
    src, dst = graph.edges()
    data_dict[('node', 'node_node', 'node')] = (src, dst)
    data_dict[('node', 'rev_node_node', 'node')] = (dst, src)
    data_dict[('node', 'node_label', 'label')] = (torch.LongTensor([]), torch.LongTensor([]))
    data_dict[('label', 'rev_node_label', 'node')] = (torch.LongTensor([]), torch.LongTensor([]))

    new_graph = dgl.heterograph(data_dict=data_dict, num_nodes_dict={'node': graph.number_of_nodes(),
                                                                     'label': num_classes})

    # copy features
    for feat in graph.ndata:
        new_graph.nodes['node'].data[feat] = graph.ndata[feat]
    # generate one_hot embeddings for label nodes
    new_graph.nodes['label'].data['feat'] = F.one_hot(torch.arange(0, num_classes), num_classes=num_classes).float()

    return new_graph


def add_connections_between_labels_nodes(graph: dgl.DGLHeteroGraph, labels: torch.Tensor, node_idx: torch.Tensor):
    """
    modify the graph structure (add connection between original nodes and label nodes) based on mask_rate
    :param graph: heterogeneous graph including original nodes and label nodes
    :param labels: Tensor, node labels
    :param node_idx: Tensor, node idx
    :return: structure-modified heterogeneous graph
    """

    graph_copy = graph.clone()

    src_indices_list, dst_indices_list = [], []

    # add connections between training nodes and label nodes
    for label_idx in labels.unique():
        # Tensor of original node idx
        original_node_idx = node_idx[labels[node_idx] == label_idx].to(labels.device)

        if len(original_node_idx) > 0:
            src_indices_list.append(original_node_idx)
            dst_indices_list.append(label_idx.repeat(len(original_node_idx)))

    src_indices = torch.cat(src_indices_list, dim=0)
    dst_indices = torch.cat(dst_indices_list, dim=0)

    # add connections between label nodes and original nodes
    graph_copy.add_edges(src_indices, dst_indices, etype=('node', 'node_label', 'label'))
    graph_copy.add_edges(dst_indices, src_indices, etype=('label', 'rev_node_label', 'node'))

    return graph_copy


def get_train_predict_truth_idx(train_idx: torch.Tensor, train_select_rate: float = 0.5):
    """
    get predict and truth indices based on mask_rate, mask on each class of the labels (guarantee each label has train and truth indices)
    :param train_idx: Tensor, train idx
    :param train_select_rate: float, training node selection rate
    :return: train_truth_idx, train_pred_idx
    """

    train_idx = train_idx[torch.randperm(train_idx.shape[0])]
    train_predict_idx, train_truth_idx = train_idx[:int(len(train_idx) * train_select_rate)], train_idx[int(len(train_idx) * train_select_rate):]

    return train_predict_idx, train_truth_idx


def get_accuracy(dataset_name: str, predicts: torch.Tensor, labels: torch.Tensor):
    """
    get accuracy for node classification
    :param dataset_name: str
    :param predicts: Tensor, shape (N, num_classes)
    :param labels: Tensor, shape (N, )
    :return:
    """

    evaluator = Evaluator(name=dataset_name)

    predictions = predicts.argmax(dim=1).cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = evaluator.eval({
        "y_true": labels.reshape(-1, 1),
        "y_pred": predictions.reshape(-1, 1)
    })['acc']

    return accuracy


def save_model_results(train_accuracy: float, val_accuracy: float, test_accuracy: float, save_result_folder: str, save_result_file_name: str):
    """
    save model result
    :param train_accuracy: float
    :param val_accuracy: float
    :param test_accuracy: float
    :param save_result_folder:
    :param save_result_file_name:
    :return:
    """
    result_json = {
        f"train accuracy": float(f"{train_accuracy:.4f}"),
        f"validate accuracy": float(f"{val_accuracy:.4f}"),
        f"test accuracy": float(f"{test_accuracy:.4f}")
    }

    result_json = json.dumps(result_json, indent=4)

    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder, exist_ok=True)
    save_result_path = os.path.join(save_result_folder, save_result_file_name)

    with open(save_result_path, 'w') as file:
        file.write(result_json)

    print(f'model result saves at {save_result_path} successfully.')
