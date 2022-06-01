import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

from utils.utils import get_accuracy


def train_model(model: nn.Module, graph: dgl.DGLGraph, node_features: dict, labels: torch.Tensor, train_idx: torch.Tensor,
                predict_category: str, loss_func: nn.Module, dataset_name: str):
    """
    train the model, return train loss, train accuracy and logits
    :param model:
    :param graph:
    :param node_features: dict
    :param labels:
    :param train_idx:
    :param predict_category: str
    :param loss_func:
    :param dataset_name:
    :return:
    """
    model.train()

    nodes_representation = model[0](graph, node_features)

    logits = model[1](nodes_representation[predict_category])

    loss = loss_func(logits[train_idx], labels[train_idx])

    train_accuracy = get_accuracy(dataset_name=dataset_name, predicts=logits[train_idx], labels=labels[train_idx])

    return loss, train_accuracy, logits


def get_pseudo_nodes_pseudo_labels(logits: torch.Tensor, evaluate_idx: torch.Tensor, loss_func: nn.Module, threshold: float, epoch: int, scale_factor: float):
    """
    get pseudo nodes, return the loss, pseudo node index and pseudo labels
    :param logits: torch.Tensor
    :param evaluate_idx: index to be evaluated
    :param loss_func:
    :param threshold: float
    :param epoch: int
    :param scale_factor: float
    :return:
    """

    training_confidence = F.sigmoid(torch.log(torch.Tensor([epoch / scale_factor]))).item()

    # only compute when the training confidence is greater than threshold
    if training_confidence > threshold:
        with torch.no_grad():

            predict_possibilities, pseudo_node_labels = F.softmax(logits[evaluate_idx], dim=1).max(dim=1)

            # training confidence
            predict_possibilities = predict_possibilities * training_confidence

            # shape (num_pseudo_nodes, )
            pseudo_node_idx = evaluate_idx[predict_possibilities > threshold]
            pseudo_node_labels = pseudo_node_labels[predict_possibilities > threshold]
            pseudo_node_predict_possibilities = predict_possibilities[predict_possibilities > threshold]

        # no pseudo nodes are selected
        if len(pseudo_node_idx) == 0:
            return 0.0, None, None, None
        else:
            # the first term logits has gradients
            # (num_pseudo_nodes, ), loss for each pseudo node
            pseudo_label_loss = loss_func(logits[pseudo_node_idx], pseudo_node_labels)

            # evaluating confidence
            pseudo_label_loss = pseudo_label_loss * pseudo_node_predict_possibilities

            pseudo_label_loss = torch.mean(pseudo_label_loss, dim=0)

            return pseudo_label_loss, pseudo_node_idx, pseudo_node_labels
    else:
        return 0.0, None, None


def evaluate_model(model: nn.Module, graph: dgl.DGLGraph, node_features: dict, labels: torch.Tensor, valid_idx: torch.Tensor, test_idx: torch.Tensor,
                   predict_category: str, loss_func: nn.Module, dataset_name: str):
    """
    evaluate the model, return validate loss, accuracy, test loss, accuracy
    :param model:
    :param graph:
    :param node_features: dict
    :param labels:
    :param valid_idx:
    :param test_idx:
    :param predict_category: str
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
    :param node_features:
    :param labels:
    :param train_idx:
    :param valid_idx:
    :param test_idx:
    :param predict_category:
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
