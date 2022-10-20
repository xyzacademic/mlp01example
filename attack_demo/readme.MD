
White box attack
=


## Prepare


Make sure the gpu device is available.

## 


## Run Example


Parameters:

 - --epsilon  Linf distance of each pixel(max `255`). Suggest `16`
   - --num-steps  PGD-attack parameters
   - --attack_type `PGD` or `FGSM`
   - --dataset  Dataset. `cifar10_binary`
   - --c0  The first class `0`
   - --c1  The second class `1`
   - --name  Output csv file location

Example:

Run

    python mlp_attack_out.py --epsilon 16 --num-steps 20 --attack_type pgd --dataset cifar10_binary --c0 0 --c1 1 --name results/outer/cifar10_binary/mlp_pgd_16_20_01


Then you will get results in terminal and a `mlp_pgd_16_20_01.csv` file at location of `results/outer/cifar10_binary/`

    gamma: 0.2
    arch: resnet18
    epsilon: 16.0
    Source model ("mlp01scd") on clean data:
    Accuracy: 0.85400, cost 0.72 seconds
    using linf PGD attack
    l2 distance:  tensor(2.9023)
    Source model ("mlp01scd") on adv data (all):
    Accuracy: 0.26900, cost 0.27 seconds
    Target model ("mlp01scd") on clean data:
    Accuracy: 0.85400, cost 0.03 seconds
    Target model ("mlp01scd") on adv data (all):
    Accuracy: 0.26900, cost 0.29 seconds




## Run Attack For Customized Model

Todo

## Run Attack for Customized Dataset

Modify your code in `tools/dataset.py`
