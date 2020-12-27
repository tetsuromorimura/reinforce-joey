This is a fork of the awesome [Joey-NMT](https://github.com/joeynmt/joeynmt) which implements Reinforcement Learning algorithms like Policy Gradient, MRT and Advantage Actor Critic.  

# &nbsp; ![Reinforce-Joey](reinforce_joey.png) Reinforce-Joey

## Implemented algorithms:  
- Policy Gradient aka REINFORCE as in [Kreutzer et al. (2017)](https://www.aclweb.org/anthology/P17-1138/)
- Minimum Risk Training as in [Shen et al. (2016)](https://www.aclweb.org/anthology/P16-1159/)
- Advantage Actor-Critic as in [Nguyen et al. (2017)](https://www.aclweb.org/anthology/D17-1153/)

You can use each algorithm with the Transformer and RNN architecture. 

## How to use 
In general cold-starting a model with Reinforcement Learning does not work too well as the methods rely on random sampling. 
That means to effectively use the algorithms you have to pretrain a Transformer or RNN and then fine-tune the model with RL. 

## Parameters
The method and hyperparameters are specified in the config, see [small.yaml](https://github.com/samukie/reinforce-joey/blob/reinforce_joey/configs/small.yaml) for an example. 

## Currently WIP: 
- clean repo  
- rework logging 
- vectorize loops  
- fix NED-A2C for Transformer
- explain hyperparameters

## Installation
Joey NMT is built on [PyTorch](https://pytorch.org/) and [torchtext](https://github.com/pytorch/text) for Python >= 3.5.

A. [*Now also directly with pip!*](https://pypi.org/project/joeynmt/)
  `pip install joeynmt`
  
B. From source
  1. Clone this repository:
  `git clone https://github.com/joeynmt/joeynmt.git`
  2. Install joeynmt and it's requirements:
  `cd joeynmt`
  `pip3 install .` (you might want to add `--user` for a local installation).
  3. Run the unit tests:
  `python3 -m unittest`

**Warning!** When running on *GPU* you need to manually install the suitable PyTorch version for your [CUDA](https://developer.nvidia.com/cuda-zone) version. This is described in the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## Reference
If you use Joey NMT in a publication or thesis, please cite the following [paper](https://arxiv.org/abs/1907.12484):

```
@inproceedings{kreutzer-etal-2019-joey,
    title = "Joey {NMT}: A Minimalist {NMT} Toolkit for Novices",
    author = "Kreutzer, Julia  and
      Bastings, Jasmijn  and
      Riezler, Stefan",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP): System Demonstrations",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-3019",
    doi = "10.18653/v1/D19-3019",
    pages = "109--114",
}
```
