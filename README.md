# &nbsp; ![Reinforce-Joey](joey-small.png) Joey NMT

## Implemented algorithms:  
- Policy Gradient aka REINFORCE as in [Kreutzer et al. (2017)](https://www.aclweb.org/anthology/P17-1138/)
- Minimum Risk Training as in [Shen et al. (2016)](https://www.aclweb.org/anthology/P16-1159/)
- Advantage Actor-Critic as in [Nguyen et al. (2017)](https://www.aclweb.org/anthology/D17-1153/)

Each algorithm is implemented for Transformer and for RNNs. 

## Currently WIP: 
- clean repo  
- rework logging  
- vectorize loops  
- fix NED-A2C for Transformer



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
