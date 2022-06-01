import torch
import torch.nn as nn
import dgl
from tqdm import tqdm
import copy
import torch.nn.functional as F

from utils.utils import convert_to_gpu, get_accuracy


def train_model(model: nn.Module, train_loader: dgl.dataloading.NodeDataLoader, labels: torch.Tensor, predict_category: str,
                loss_func: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, dataset_name: str, device: str):
    """
    train the model, return train loss, train accuracy
    :param model:
    :param train_loader:
    :param labels:
    :param predict_category: str
    :param loss_func:
    :param optimizer:
    :param scheduler:
    :param dataset_name:
    :param device:
    :return:
    """
    model.train()

    train_y_trues = []
    train_y_predicts = []
    train_total_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, ncols=120)

    for batch, (input_nodes, output_nodes, blocks) in enumerate(train_loader_tqdm):
        blocks = [convert_to_gpu(b, device=device) for b in blocks]

        nodes_representation = model[0](blocks, copy.deepcopy({ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in input_nodes.keys()}))

        train_y_predict = model[1](nodes_representation[predict_category])

        train_y_true = convert_to_gpu(labels[output_nodes[predict_category]], device=device)

        loss = loss_func(train_y_predict, train_y_true)

        train_total_loss += loss.item()

        train_y_predicts.append(train_y_predict.detach().cpu())
        train_y_trues.append(train_y_true.detach().cpu())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loader_tqdm.set_description(f'train for the {batch + 1}-th batch, train loss: {loss.item()}')

        # step should be called after a batch has been used for training.
        scheduler.step()

    train_total_loss /= (batch + 1)
    train_y_predicts = torch.cat(train_y_predicts, dim=0)
    train_y_trues = torch.cat(train_y_trues, dim=0)

    train_accuracy = get_accuracy(dataset_name=dataset_name, predicts=train_y_predicts, labels=train_y_trues)

    return train_total_loss, train_accuracy


def get_pseudo_nodes_pseudo_labels(model: nn.Module, train_loader: dgl.dataloading.NodeDataLoader, threshold: float, epoch: int,
                                   scale_factor: float, predict_category: str, device: str):
    """
    get pseudo nodes, return the loss, pseudo node index and pseudo labels

    :param model:
    :param train_loader:
    :param threshold: float
    :param epoch:
    :param scale_factor: float
    :param predict_category:
    :param device:
    :return:
    """
    pseudo_node_idx_list = []
    pseudo_node_labels_list = []
    pseudo_node_predict_possibilities_list = []

    training_confidence = F.sigmoid(torch.log(torch.Tensor([epoch / scale_factor]))).item()

    # only compute when the training confidence is greater than threshold
    if training_confidence > threshold:
        train_loader_tqdm = tqdm(train_loader, ncols=120)

        with torch.no_grad():
            for batch, (input_nodes, output_nodes, blocks) in enumerate(train_loader_tqdm):

                blocks = [convert_to_gpu(b, device=device) for b in blocks]

                nodes_representation = model[0](blocks, copy.deepcopy(
                    {ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in input_nodes.keys()}))

                train_y_predict = model[1](nodes_representation[predict_category])

                predict_possibilities, pseudo_node_labels = F.softmax(train_y_predict, dim=1).max(dim=1)

                # training confidence
                predict_possibilities = predict_possibilities * training_confidence

                # shape (num_pseudo_nodes, )
                pseudo_node_idx = output_nodes[predict_category][predict_possibilities > threshold]
                pseudo_node_labels = pseudo_node_labels[predict_possibilities > threshold]
                pseudo_node_predict_possibilities = predict_possibilities[predict_possibilities > threshold]

                train_loader_tqdm.set_description(f'get pseudo nodes for the {batch + 1}-th batch')

                # no pseudo nodes are selected
                if len(pseudo_node_idx) == 0:
                    continue

                else:
                    pseudo_node_idx_list.append(pseudo_node_idx.cpu())
                    pseudo_node_labels_list.append(pseudo_node_labels.cpu())
                    pseudo_node_predict_possibilities_list.append(pseudo_node_predict_possibilities)

    if len(pseudo_node_idx_list) > 0 and len(pseudo_node_labels_list) > 0:
        return torch.cat(pseudo_node_idx_list, dim=0), torch.cat(pseudo_node_labels_list, dim=0), \
               torch.cat(pseudo_node_predict_possibilities_list, dim=0)
    else:
        return None, None, None


def train_on_pseudo_nodes(model: nn.Module, train_loader: dgl.dataloading.NodeDataLoader, pseudo_node_idx: torch.Tensor,
                          pseudo_node_labels: torch.Tensor, labels: torch.Tensor, pseudo_node_predict_possibilities: torch.Tensor, balance_factor: float,
                          predict_category: str, loss_func: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, device: str):
    """
    train on pseudo nodes

    :param model:
    :param train_loader:
    :param pseudo_node_idx: (pseudo_nodes_num, )
    :param pseudo_node_labels: (pseudo_nodes_num, )
    :param labels: (nodes_num, )
    :param pseudo_node_predict_possibilities: (pseudo_nodes_num, )
    :param balance_factor: float
    :param predict_category:
    :param loss_func:
    :param optimizer:
    :param scheduler:
    :param device:
    :return:
    """
    labels[pseudo_node_idx] = pseudo_node_labels
    node_predict_possibilities = torch.zeros(len(labels)).to(device)
    node_predict_possibilities[pseudo_node_idx] = pseudo_node_predict_possibilities
    train_loader_tqdm = tqdm(train_loader, ncols=120)

    for batch, (input_nodes, output_nodes, blocks) in enumerate(train_loader_tqdm):
        blocks = [convert_to_gpu(b, device=device) for b in blocks]

        nodes_representation = model[0](blocks, copy.deepcopy(
            {ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in input_nodes.keys()}))

        train_y_predict = model[1](nodes_representation[predict_category])

        train_y_true = convert_to_gpu(labels[output_nodes[predict_category]], device=device)

        pseudo_label_loss = loss_func(train_y_predict, train_y_true)

        # evaluating confidence
        pseudo_label_loss = pseudo_label_loss * node_predict_possibilities[output_nodes[predict_category]]

        # Tensor, shape (num_evaluate_nodes, )
        pseudo_label_loss = balance_factor * torch.mean(pseudo_label_loss, dim=0)
        optimizer.zero_grad()
        pseudo_label_loss.backward()
        optimizer.step()

        # step should be called after a batch has been used for training.
        scheduler.step()

        train_loader_tqdm.set_description(f'train on pseudo nodes for the {batch + 1}-th batch')


def evaluate_model(model: nn.Module, loader: dgl.dataloading.NodeDataLoader, labels: torch.Tensor, predict_category: str,
                   loss_func: nn.Module, dataset_name: str, device: str, mode: str):
    """
    evaluate the model, return evaluate loss, accuracy
    :param model:
    :param loader:
    :param labels:
    :param predict_category:
    :param loss_func:
    :param dataset_name:
    :param device:
    :param mode:
    :return:
    """
    model.eval()

    with torch.no_grad():
        y_trues = []
        y_predicts = []
        total_loss = 0.0
        loader_tqdm = tqdm(loader, ncols=120)

        for batch, (input_nodes, output_nodes, blocks) in enumerate(loader_tqdm):
            blocks = [convert_to_gpu(b, device=device) for b in blocks]

            nodes_representation = model[0](blocks, copy.deepcopy({ntype: blocks[0].srcnodes[ntype].data['feat'] for ntype in input_nodes.keys()}))

            y_predict = model[1](nodes_representation[predict_category])
            # Tensor, (samples_num, )
            y_true = convert_to_gpu(labels[output_nodes[predict_category]], device=device)

            loss = loss_func(y_predict, y_true)

            total_loss += loss.item()

            y_predicts.append(y_predict.detach().cpu())
            y_trues.append(y_true.detach().cpu())

            loader_tqdm.set_description(f'{mode} for the {batch + 1}-th batch, {mode} loss: {loss.item()}')

        total_loss /= (batch + 1)
        y_predicts = torch.cat(y_predicts, dim=0)
        y_trues = torch.cat(y_trues, dim=0)

        accuracy = get_accuracy(dataset_name=dataset_name, predicts=y_predicts, labels=y_trues)

    return total_loss, accuracy


def get_final_performance(model: nn.Module, graph: dgl.DGLGraph, labels: torch.Tensor, predict_category: str, train_idx: torch.Tensor,
                          valid_idx: torch.Tensor, test_idx: torch.Tensor, dataset_name: str, device: str):
    """

    :param model:
    :param graph:
    :param labels:
    :param predict_category:
    :param train_idx:
    :param valid_idx:
    :param test_idx:
    :param dataset_name:
    :param device:
    :return:
    """
    model.eval()

    with torch.no_grad():
        nodes_representation = model[0].inference(graph, copy.deepcopy(
            {ntype: graph.nodes[ntype].data['feat'] for ntype in graph.ntypes}), device=device)

        train_y_predicts = model[1](convert_to_gpu(nodes_representation[predict_category], device=device))[train_idx]
        train_y_trues = convert_to_gpu(labels[train_idx], device=device)
        train_accuracy = get_accuracy(dataset_name=dataset_name, predicts=train_y_predicts, labels=train_y_trues)

        val_y_predicts = model[1](convert_to_gpu(nodes_representation[predict_category], device=device))[valid_idx]
        val_y_trues = convert_to_gpu(labels[valid_idx], device=device)
        val_accuracy = get_accuracy(dataset_name=dataset_name, predicts=val_y_predicts, labels=val_y_trues)

        test_y_predicts = model[1](convert_to_gpu(nodes_representation[predict_category], device=device))[test_idx]
        test_y_trues = convert_to_gpu(labels[test_idx], device=device)
        test_accuracy = get_accuracy(dataset_name=dataset_name, predicts=test_y_predicts, labels=test_y_trues)

    return train_accuracy, val_accuracy, test_accuracy
