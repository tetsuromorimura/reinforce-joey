# &nbsp; ![Reinforce-Joey](reinforce_joey.png) Reinforce-Joey
This is a fork of the awesome Joey-NMT which contains the implementations of several Reinforcement Learning algorithms, reward functions and baselines.

## Implemented algorithms:  
The forward pass of each method can be found here:  
[REINFORCE](https://github.com/samukie/reinforce-joey/blob/b10f93314ccc9f3994e38b21c4b1ed21519747cc/joeynmt/model.py#L80), [MRT](https://github.com/samukie/reinforce-joey/blob/b10f93314ccc9f3994e38b21c4b1ed21519747cc/joeynmt/model.py#L152), [NED_A2C](https://github.com/samukie/reinforce-joey/blob/b10f93314ccc9f3994e38b21c4b1ed21519747cc/joeynmt/model.py#L255)   
 
 The implemented papers can be found here:  
 
- Policy Gradient aka REINFORCE as in [Kreutzer et al. (2017)](https://www.aclweb.org/anthology/P17-1138/)
- Minimum Risk Training as in [Shen et al. (2016)](https://www.aclweb.org/anthology/P16-1159/)
- Advantage Actor-Critic aka NED-A2C as in [Nguyen et al. (2017)](https://www.aclweb.org/anthology/D17-1153/)

Each algorithm can be used with the Transformer and the RNN architecture. 

## How to use 
In general cold-starting a model with Reinforcement Learning does not work too well as the methods rely on random sampling. 
That means to effectively use the algorithms you have to pretrain a Transformer/RNN or download a [pretrained model](https://github.com/joeynmt/joeynmt/blob/master/README.md#pre-trained-models) and then fine-tune with RL. 

## Parameters
The method and hyperparameters are specified in the config, see [small.yaml](https://github.com/samukie/reinforce-joey/blob/reinforce_joey/configs/small.yaml) for an example. 
Here a short explanation of the parameters.  
* All methods: 
  * temperature: Softmax temperature parameter. Decreasing results in a 'peakier' distribution and more exploitation while increasing leads to more exploration.  
* Policy Gradient/Reinforce:   
  * reward: 
    * bleu: standard corpus_bleu from sacrebleu
    * scaled_bleu: scales the BLEU locally to the interval [-0.5, 0.5] for each batch
    * constant: constant reward of 1 
  * baseline: 
    * False: no baseline
    * average_reward_baseline: subtracts a running average of all previous BLEUs from the rewards
* MRT:  
  * add_gold: adds gold/reference to sample space
  * samples: number of samples
  * alpha: mrt smoothness parameter 
  Advantage Actor-Critic:  
* critic_learning_rate: learning rate of critic network

## Currently WIP: 
- fix NED-A2C for Transformer
- improve logging/add more options 
- add learned baseline parameters to config

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

## Acknowledgements
Thanks to [Michael Staniek](https://github.com/MStaniek) who was a great help with the implementations

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
