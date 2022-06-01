import torch
import torch.nn as nn
import dgl

from utils.utils import get_accuracy


def train_model(model: nn.Module, graph: dgl.DGLGraph, node_features: dict, labels: torch.Tensor, train_idx: torch.Tensor, predict_category: str,
                loss_func: nn.Module, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, dataset_name: str):
    """
    train the model, return train loss, train accuracy
    :param model:
    :param graph:
    :param node_features: dict
    :param labels:
    :param train_idx:
    :param predict_category: str
    :param loss_func:
    :param optimizer:
    :param scheduler:
    :param dataset_name:
    :return:
    """
    model.train()

    nodes_representation = model[0](graph, node_features)

    logits = model[1](nodes_representation[predict_category])

    loss = loss_func(logits[train_idx], labels[train_idx])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # step should be called after a batch has been used for training.
    scheduler.step()

    train_accuracy = get_accuracy(dataset_name=dataset_name, predicts=logits[train_idx], labels=labels[train_idx])

    return loss.item(), train_accuracy


def evaluate_model(model: nn.Module, graph: dgl.DGLGraph, node_features: dict, labels: torch.Tensor, valid_idx: torch.Tensor, test_idx: torch.Tensor,
                   predict_category: str, loss_func: nn.Module, dataset_name: str):
    """
    evaluate the model, return validate loss, metric, test loss, metric
    :param model:
    :param graph:
    :param predict_category: str
    :param labels:
    :param valid_idx:
    :param test_idx:
    :param node_features: dict
    :param loss_func:
    :param dataset_name:
    :return:
    """
    model.eval()

    with torch.no_grad():

        nodes_representation = model[0](graph, node_features)

        logits = model[1](nodes_representation[predict_category])

        val_loss = loss_func(logits[valid_idx], labels[valid_idx])
        val_accuracy = get_accuracy(dataset_name=dataset_name, predicts=logits[valid_idx], labels=labels[valid_idx])

        test_loss = loss_func(logits[test_idx], labels[test_idx])
        test_accuracy = get_accuracy(dataset_name=dataset_name, predicts=logits[test_idx], labels=labels[test_idx])

    return val_loss.item(), val_accuracy, test_loss.item(), test_accuracy


def get_final_performance(model: nn.Module, graph: dgl.DGLGraph, node_features: dict, labels: torch.Tensor, train_idx: torch.Tensor, valid_idx: torch.Tensor,
                          test_idx: torch.Tensor, predict_category: str, dataset_name: str):
    """

    :param model:
    :param graph:
    :param predict_category: str
    :param labels:
    :param train_idx:
    :param valid_idx:
    :param test_idx:
    :param node_features: dict
    :param dataset_name:
    :return:
    """
    # evaluate the best model
    model.eval()

    with torch.no_grad():

        nodes_representation = model[0](graph, node_features)

        logits = model[1](nodes_representation[predict_category])

        train_accuracy = get_accuracy(dataset_name=dataset_name, predicts=logits[train_idx], labels=labels[train_idx])

        val_accuracy = get_accuracy(dataset_name=dataset_name, predicts=logits[valid_idx], labels=labels[valid_idx])

        test_accuracy = get_accuracy(dataset_name=dataset_name, predicts=logits[test_idx], labels=labels[test_idx])

    return train_accuracy, val_accuracy, test_accuracy
