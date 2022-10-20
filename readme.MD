
# MLP01SCD Tiny Demo code

## Reference
1. [Gradient free sign activation zero one loss neural networks for adversarially robust classification](https://digitalcommons.njit.edu/dissertations/1545/)

2. [Paper code for reproducing results in the paper](https://github.com/xyzacademic/scd_deep)


3. [Full code for usage](https://github.com/zero-one-loss/scd)


## Prepare

1. Install pytorch
```shell
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

   


2. Install advertorch

```shell
pip install git+https://github.com/BorealisAI/advertorch.git
```
    

3. You may also need `sklearn` and `pandas` as well.



## Guide

1. Training neural network is in `main`.
2. Run for `PGD/FGSM` attack in `attack_demo`.