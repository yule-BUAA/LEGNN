# Label-Enhanced Graph Neural Network for Semi-supervised Node Classification

The experiments are conducted on the ogbn-arxiv and ogbn-mag datasets on the Stanford OGB (1.3.2) benchmark. 
The description of "Label-Enhanced Graph Neural Network for Semi-supervised Node Classification" is [available here](http://arxiv.org/abs/2205.15653). 

### Datasets:
We provide the preprocessed datasets at [here](https://drive.google.com/file/d/1WNbjmxblF1QNbT9IuZRspoqOifJiwvwS/view?usp=sharing), 
which should be put in the ```./dataset``` folder.

### To run the node classification task on ogbn-arxiv:
  - run ```./train_LEGNN/train_full_batch_LEGNN.py``` to get the results of LEGNN on ogbn-arxiv.
  - run ```./train_LEGNN_ASTrain/train_full_batch_LEGNN_ASTrain.py``` to get the results of LEGNN + AS-Train on ogbn-arxiv.

### To run the node classification task on ogbn-mag:
  1. in addition to downloading the preprocessed datasets, you could also run ```./preprocess_data/preprocess_ogbn_mag.py``` to preprocess the original ogbn-mag dataset. 
  2.   
  - run ```./train_LEGNN/train_mini_batch_RGNN.py``` to get the results of LEGNN on ogbn-mag.
  - run ```./train_LEGNN_ASTrain/train_mini_batch_LEGNN_ASTrain.py``` to get the results of LEGNN + AS-Train on ogbn-mag.

### Performance on ogbn-arxiv
| Model        | Test Accuracy   | Valid Accuracy  | # Parameter     | Hardware         |
| ---------    | --------------- | --------------  | --------------  |--------------    |
| LEGNN  | 0.7337 ± 0.0007   | 0.7480 ± 0.0009  |    5,374,120      | NVIDIA Tesla T4 (15 GB) |
| LEGNN + AS-Train  | 0.7371 ± 0.0011   | 0.7494 ± 0.0008  |    5,374,120      | NVIDIA Tesla T4 (15 GB) |

### Performance on ogbn-mag
| Model        | Test Accuracy   | Valid Accuracy  | # Parameter     | Hardware         |
| ---------    | --------------- | --------------  | --------------  |--------------    |
| LEGNN  | 0.5276 ± 0.0014   | 0.5443 ± 0.0009  |    5,147,997      | NVIDIA Tesla T4 (15 GB) |
| LEGNN + AS-Train  | 0.5378 ± 0.0016  | 0.5528 ± 0.0013  |    5,147,997      | NVIDIA Tesla T4 (15 GB) |

## Environments:
- [PyTorch 1.8.1](https://pytorch.org/)
- [DGL 0.7.0](https://www.dgl.ai/)
- [PyTorch Geometric 2.0.1](https://pytorch-geometric.readthedocs.io/en/latest/)
- [OGB 1.3.2](https://ogb.stanford.edu/docs/home/)
- [tqdm](https://github.com/tqdm/tqdm)
- [numpy](https://github.com/numpy/numpy)


## Citation
Please consider citing our paper when using the codes.

```bibtex
@article{yu2022label,
  title={Label-Enhanced Graph Neural Network for Semi-supervised Node Classification},
  author={Yu, Le and Sun, Leilei and Du, Bowen and Zhu, Tongyu and Lv, Weifeng},
  journal={arXiv preprint arXiv:2205.15653},
  year={2022}
}
```
