#!/bin/bash

for (( rd=0; rd <8; rd++ ))

do
python train_cnn01_01.py --nrows 0.75 --localit 1 --updated_fc_features 128 --updated_fc_nodes 1 \
--width 100 --normalize 1 --percentile 1 --fail_count 1 --loss 01loss --act sign --fc_diversity 1 \
--init normal --no_bias 0 --scale 1 --w-inc1 0.17 --w-inc2 0.17 --version mlp01scale --seed 0 \
--iters 1000 --dataset cifar10_binary --n_classes 2 --cnn 0 --divmean 0 --target cifar10_binary_01_mlp01scd_0 \
--updated_fc_ratio 1 --verbose_iter 50 --c0 0 --c1 1

done