## R-DANN Transfer Technique

This is adapted from the pytorch implementation of the paper _[Unsupervised Domain Adaptation by Backpropagation](http://sites.skoltech.ru/compvision/projects/grl/)_ in the repository [fungtion/DANN.git](https://github.com/fungtion/DANN.git)

Although R-DANN was not discussed as a technique being benchmarked in the paper, this transfer learning technique was also explored. Originally run separately, the R-DANN logic has been included as a subdirectory in this repository to centralize the code for this paper.

The R-DANN technique was explored separate from the remaining methods, so this subdirectory is also made capable of running as an independent unit in order to reflect that. The ESB preprocessed data has been copied into it to enable a sample run. The code is run from [main.py](./main.py).
