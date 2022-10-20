import torch
import os
import time
import numpy as np
import sys
import torch.backends.cudnn as cudnn
from sklearn.metrics import balanced_accuracy_score, accuracy_score
sys.path.append('..')

from tools import args, load_data, ModelArgs, BalancedBatchSampler, get_set
from core.lossfunction import *
from core.cnn01 import *

from core.train_cnn01_basic_01_ import train_single_cnn01
import pickle
import pandas as pd
# Args assignment
scd_args = ModelArgs()

scd_args.nrows = args.nrows
scd_args.local_iter = args.localit
scd_args.num_iters = args.iters
scd_args.interval = args.interval
scd_args.rounds = 1
scd_args.w_inc1 = args.w_inc1
scd_args.updated_fc_features = args.updated_fc_features
scd_args.updated_conv_features = args.updated_conv_features
scd_args.n_jobs = args.n_jobs
scd_args.num_gpus = 1
scd_args.adv_train = bool(args.adv_train)
scd_args.eps = args.eps
scd_args.w_inc2 = args.w_inc2
scd_args.hidden_nodes = 20
scd_args.evaluation = True
scd_args.verbose = True
scd_args.b_ratio = 0.2
scd_args.cuda = True if torch.cuda.is_available() else False
scd_args.seed = args.seed
scd_args.source = args.source
scd_args.save = True
scd_args.resume = True if args.resume else False
scd_args.loss = args.loss
scd_args.criterion = criterion[scd_args.loss]
scd_args.structure = arch[args.version]
scd_args.dataset = args.dataset
scd_args.num_classes = args.n_classes
scd_args.gpu = 0
scd_args.fp16 = True if args.fp16 and scd_args.cuda else False
scd_args.act = args.act
scd_args.updated_fc_nodes = args.updated_fc_nodes
scd_args.updated_conv_nodes = args.updated_conv_nodes
scd_args.width = args.width
scd_args.normal_noise = True if args.normal_noise else False
scd_args.verbose = True
scd_args.normalize = bool(args.normalize)
scd_args.batch_size = 256
scd_args.sigmoid = True if scd_args.loss == 'bce' else False
scd_args.softmax = True if scd_args.loss == 'mce' else False
scd_args.percentile = bool(args.percentile)
scd_args.fail_count = args.fail_count
scd_args.diversity = False
scd_args.fc_diversity = bool(args.fc_diversity)
scd_args.conv_diversity = False
scd_args.updated_conv_features_diversity = 16
scd_args.diversity_train_stop_iters = 3000
scd_args.init = args.init
scd_args.target = args.target
scd_args.logs = {}
scd_args.no_bias = args.no_bias
scd_args.record = False
scd_args.scale = args.scale
scd_args.save_path = os.path.join('checkpoints', 'pt')
scd_args.adaptive_loss_epoch = args.epoch
scd_args.updated_fc_ratio = args.updated_fc_ratio
scd_args.updated_conv_ratio = args.updated_conv_ratio
scd_args.divmean = args.divmean
scd_args.cnn = args.cnn
scd_args.verbose_iter = args.verbose_iter
scd_args.freeze_layer = args.freeze_layer
scd_args.temp_save_per_iter = args.temp_save_per_iter
scd_args.lr_decay_iter = args.lr_decay_iter
scd_args.batch_increase_iter = args.batch_increase_iter
scd_args.aug = args.aug
scd_args.balanced_sampling = args.balanced_sampling
scd_args.bnn_layer = args.bnn_layer
scd_args.epsilon = args.epsilon
scd_args.step = args.steps
scd_args.alpha = args.alpha
scd_args.bp_layer = args.bp_layer
scd_args.lr = args.lr
scd_args.reinit = args.reinit
scd_args.bnn_layers = args.bnn_layers
scd_args.pnorm = args.pnorm
scd_args.adv_source = args.adv_source
scd_args.adv_structure = arch[args.adv_structure] if scd_args.adv_source is not None else None
np.random.seed(scd_args.seed)

print(f'c0: {args.c0} c1: {args.c1}')
train_data, test_data, train_label, test_label = load_data(args.dataset, args.n_classes,
                                                           c1=args.c0, c2=args.c1)

# normal_noise = True if args.normal_noise else False
# if normal_noise:
#     print(f'Normal noise augmented with eps: {args.epsilon}')
#     noise = np.random.normal(0, 1, size=train_data.shape)
#     noisy = np.clip((train_data + noise * args.epsilon), 0, 1)
#     train_data = np.concatenate([train_data, noisy], axis=0)
#     train_label = np.concatenate([train_label] * 2, axis=0)
#     print(f'train data shape: {train_data.shape}')

if scd_args.cnn:
    if 'cifar10' in scd_args.dataset:
        train_data = train_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
        test_data = test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
    elif 'stl10' in scd_args.dataset:
        train_data = train_data.reshape((-1, 96, 96, 3)).transpose((0, 3, 1, 2))
        test_data = test_data.reshape((-1, 96, 96, 3)).transpose((0, 3, 1, 2))
    elif 'celeba' in scd_args.dataset:
        train_data = train_data.reshape((-1, 3, 96, 96))
        test_data = test_data.reshape((-1, 3, 96, 96))
    elif 'gtsrb' in scd_args.dataset:
        train_data = train_data.reshape((-1, 3, 48, 48))
        test_data = test_data.reshape((-1, 3, 48, 48))

if scd_args.divmean:
    train_data = train_data / 0.5 - 1
    test_data = test_data / 0.5 - 1
np.random.seed(scd_args.seed)

trainset_, testset_ = get_set(data=scd_args.dataset, aug=scd_args.aug,
                              num_classes=scd_args.num_classes, c0=args.c0, c1=args.c1)

best_model, val_acc = train_single_cnn01(scd_args, None, None,
(train_data, test_data, train_label, test_label), trainset_)