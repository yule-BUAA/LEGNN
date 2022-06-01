import torch
import torch.nn as nn
import warnings
import os
import shutil
import copy
import logging
import time

import sys
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
path_project = os.path.split(root_path)[0]
sys.path.append(root_path)
sys.path.append(path_project)

from utils.utils import set_random_seed, convert_to_gpu, get_n_params, get_optimizer, get_lr_scheduler, load_dataset, save_model_results, \
    get_node_data_loader, generate_hetero_graph, add_connections_between_labels_nodes, get_train_predict_truth_idx, get_loss_func
from utils.mini_batch_ASTrain_utils import train_model, evaluate_model, get_final_performance, get_pseudo_nodes_pseudo_labels, train_on_pseudo_nodes
from model.LEGNN import LEGAT
from model.Classifier import Classifier
from utils.EarlyStopping import EarlyStopping

args = {
    'gnn_model_name': 'GAT',
    'dataset': 'ogbn-mag',
    'predict_category': 'node',
    'seed': 0,
    'cuda': 0,
    'batch_size': 370,
    'node_neighbors_min_num': 15,
    'learning_rate': 0.001,
    'hidden_units': [64, 64, 64],
    'num_heads': 8,
    'input_drop': 0.0,
    'feat_drop': 0.1,
    'output_drop': 0.3,
    'use_attn_dst': True,
    'use_symmetric_norm': False,
    'residual': True,
    'norm': False,
    'train_select_rate': 0.5,
    'lr_t_max': 50000,
    'threshold': 0.6,
    'scale_factor': 5.0,
    'balance_factor': 0.1,
    'optimizer': 'adam',
    'weight_decay': 0,
    'epochs': 500,
    'patience': 50,
    'print_test_interval': 10,
    'num_runs': 10
}
args['device'] = f'cuda:{args["cuda"]}' if torch.cuda.is_available() and args["cuda"] >= 0 else 'cpu'

if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    train_accuracy_list, val_accuracy_list, test_accuracy_list = [], [], []

    for run in range(args['num_runs']):
        args['seed'] = run
        args['model_name'] = f'LEGNN_ASTrain_seed{args["seed"]}'

        set_random_seed(args['seed'])

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args['dataset']}/{args['model_name']}", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(f"./logs/{args['dataset']}/{args['model_name']}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        graph, labels, num_classes, train_idx, valid_idx, test_idx = load_dataset(root_path='../dataset', dataset_name=args['dataset'])

        graph = generate_hetero_graph(graph, num_classes)
        # Create csr/coo/csc formats before launching sampling processes
        # This avoids creating certain formats in each data loader process, which saves memory and CPU.
        graph.create_formats_()

        evaluate_graph = add_connections_between_labels_nodes(graph, labels, train_idx)
        evaluate_graph.create_formats_()

        _, val_loader, test_loader = get_node_data_loader(args['node_neighbors_min_num'],
                                                          len(args['hidden_units']),
                                                          evaluate_graph,
                                                          batch_size=args['batch_size'],
                                                          train_idx=train_idx,
                                                          valid_idx=valid_idx,
                                                          test_idx=test_idx,
                                                          sampled_node_type=args['predict_category'],
                                                          num_workers=4)
        legat = LEGAT(input_dim_dict={ntype: graph.nodes[ntype].data['feat'].shape[1] for ntype in graph.ntypes}, hidden_sizes=args['hidden_units'], etypes=graph.etypes,
                      ntypes=graph.ntypes, num_heads=args['num_heads'], residual=args['residual'], input_drop=args['input_drop'], feat_drop=args['feat_drop'],
                      output_drop=args['output_drop'], use_attn_dst=args['use_attn_dst'], norm=args['norm'], use_symmetric_norm=args['use_symmetric_norm'], full_batch=False)

        classifier = Classifier(n_hid=args['hidden_units'][-1] * args['num_heads'], n_out=num_classes)

        model = nn.Sequential(legat, classifier)

        model = convert_to_gpu(model, device=args['device'])
        logger.info(model)

        logger.info(f'the size of LEGNN_ASTrain parameters is {get_n_params(model)}.')

        logger.info(f'configuration is {args}')
        optimizer = get_optimizer(model, args['optimizer'], args['learning_rate'], args['weight_decay'])

        scheduler = get_lr_scheduler(optimizer, learning_rate=args['learning_rate'], t_max=args['lr_t_max'])

        model = convert_to_gpu(model, device=args['device'])

        save_model_folder = f"./save_model/{args['dataset']}/{args['model_name']}"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args['patience'], save_model_folder=save_model_folder,
                                       save_model_name=args['model_name'], logger=logger)

        loss_func = get_loss_func(dataset_name=args['dataset'])
        evaluate_loss_func = get_loss_func(dataset_name=args['dataset'], reduction='none')

        for epoch in range(args['epochs']):

            # compute the loss for training nodes, so get train predict and truth idx based on train_idx
            train_predict_idx, train_truth_idx = get_train_predict_truth_idx(train_idx, args['train_select_rate'])

            train_graph = add_connections_between_labels_nodes(graph, labels, train_truth_idx)
            train_graph.create_formats_()

            train_loader, _, _ = get_node_data_loader(args['node_neighbors_min_num'],
                                                      len(args['hidden_units']),
                                                      train_graph,
                                                      batch_size=args['batch_size'],
                                                      train_idx=train_predict_idx,
                                                      valid_idx=valid_idx,
                                                      test_idx=test_idx,
                                                      sampled_node_type=args['predict_category'],
                                                      num_workers=4)

            train_loss, train_accuracy = train_model(model, train_loader, labels, args['predict_category'], loss_func, optimizer,
                                                     scheduler, dataset_name=args['dataset'], device=args['device'])

            train_loader, _, _ = get_node_data_loader(args['node_neighbors_min_num'],
                                                      len(args['hidden_units']),
                                                      train_graph,
                                                      batch_size=args['batch_size'],
                                                      train_idx=torch.cat([valid_idx, test_idx], dim=0),
                                                      valid_idx=valid_idx,
                                                      test_idx=test_idx,
                                                      sampled_node_type=args['predict_category'],
                                                      num_workers=4)
            # get pseudo nodes, not backward
            pseudo_node_idx, pseudo_node_labels, pseudo_node_predict_possibilities = get_pseudo_nodes_pseudo_labels(model, train_loader,
                                                                                                                    threshold=args['threshold'],
                                                                                                                    epoch=epoch + 1,
                                                                                                                    scale_factor=args['scale_factor'],
                                                                                                                    predict_category=args['predict_category'],
                                                                                                                    device=args['device'])
            # if exist pseudo nodes, then backward
            if pseudo_node_idx is not None and len(pseudo_node_idx) > 0:
                train_loader, _, _ = get_node_data_loader(args['node_neighbors_min_num'],
                                                          len(args['hidden_units']),
                                                          train_graph,
                                                          batch_size=args['batch_size'],
                                                          train_idx=pseudo_node_idx,
                                                          valid_idx=valid_idx,
                                                          test_idx=test_idx,
                                                          sampled_node_type=args['predict_category'],
                                                          num_workers=4)

                train_on_pseudo_nodes(model, train_loader, pseudo_node_idx=pseudo_node_idx, pseudo_node_labels=pseudo_node_labels, labels=copy.deepcopy(labels),
                                      pseudo_node_predict_possibilities=pseudo_node_predict_possibilities, balance_factor=args['balance_factor'],
                                      predict_category=args['predict_category'], loss_func=evaluate_loss_func,
                                      optimizer=optimizer, scheduler=scheduler, device=args['device'])

            val_loss, val_accuracy = evaluate_model(model, val_loader, labels, args['predict_category'], loss_func, dataset_name=args['dataset'],
                                                    device=args['device'], mode='validate')

            if (epoch + 1) % args['print_test_interval'] == 0:
                test_loss, test_accuracy = evaluate_model(model, test_loader, labels, args['predict_category'], loss_func,
                                                          dataset_name=args['dataset'], device=args['device'], mode='test')
                logger.info(
                    f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_loss:.4f}, accuracy {train_accuracy:.4f}, '
                    f'valid loss: {val_loss:.4f}, accuracy {val_accuracy:.4f}, test loss: {test_loss:.4f}, accuracy {test_accuracy:.4f}')
            else:
                logger.info(
                    f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {train_loss:.4f}, accuracy {train_accuracy:.4f}, '
                    f'valid loss: {val_loss:.4f}, accuracy {val_accuracy:.4f}')

            early_stop = early_stopping.step([('accuracy', val_accuracy, True)], model)
            if early_stop:
                break

        # load best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info('get final performance...')

        # convert to cpu to avoid the out of memory of cuda
        model = model.cpu()
        args['device'] = 'cpu'

        # evaluate the best model
        train_accuracy, val_accuracy, test_accuracy = get_final_performance(model, evaluate_graph, labels, args['predict_category'], train_idx,
                                                                            valid_idx, test_idx, dataset_name=args['dataset'], device=args['device'])

        logger.info(
            f'final train accuracy {train_accuracy:.4f}, valid accuracy {val_accuracy:.4f}, test accuracy {test_accuracy:.4f}')


        save_model_results(train_accuracy, val_accuracy, test_accuracy, save_result_folder=f"./results/{args['dataset']}",
                           save_result_file_name=f"{args['model_name']}.json")

        train_accuracy_list.append(train_accuracy)
        val_accuracy_list.append(val_accuracy)
        test_accuracy_list.append(test_accuracy)

        if run < args['num_runs'] - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

    logger.info(f'train accuracy: {train_accuracy_list}')
    logger.info(f'val accuracy: {val_accuracy_list}')
    logger.info(f'test accuracy: {test_accuracy_list}')
    logger.info(f'average train accuracy: {torch.mean(torch.Tensor(train_accuracy_list)):.4f} ± {torch.std(torch.Tensor(train_accuracy_list)):.4f}')
    logger.info(f'average val accuracy: {torch.mean(torch.Tensor(val_accuracy_list)):.4f} ± {torch.std(torch.Tensor(val_accuracy_list)):.4f}')
    logger.info(f'average test accuracy: {torch.mean(torch.Tensor(test_accuracy_list)):.4f} ± {torch.std(torch.Tensor(test_accuracy_list)):.4f}')
