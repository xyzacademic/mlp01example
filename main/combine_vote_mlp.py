import sys
sys.path.append('..')
import os
from tools import args, load_data, ModelArgs, BalancedBatchSampler, print_title
import pickle
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import torch
import numpy as np
from core.ensemble_model import *
import time


if __name__ == '__main__':
    path = 'checkpoints/pt'
    # args.votes = 2
    # args.target = 'cifar10_toy2_relu_i1_mce_nb1_nw0_dm1_s2_fp32_32'
    # args.version = 'toy2'
    # args.scale = 1
    # args.dataset = 'cifar10'
    # args.n_classes = 10
    # args.act = 'relu'
    # args.cnn = 1
    checkpoints = [os.path.join(path, f'{args.target}_%d.pt' % i) for i in range(args.votes)]
    state_dict = [torch.load(checkpoints[i], map_location=torch.device('cpu')) for i in range(len(checkpoints))]

    structure = arch[args.version]
    params = {
        'num_classes': 1,
        'votes': args.votes,
        'scale': args.scale,
        'act': args.act,
        'sigmoid': True if 'bce' in args.target else False,
        'softmax': False,
        'bias': bool(1-args.no_bias),
    }

    scd = structure(**params)
    layers = scd.layers

    for layer in layers:
        if 'conv' in layer:
            weights = torch.cat([state_dict[i][f'{layer}.weight'] for i in range(args.votes)])
            scd._modules[layer].weight = torch.nn.Parameter(weights, requires_grad=False)
            try:
                bias = torch.cat([state_dict[i][f'{layer}.bias'] for i in range(args.votes)])
                scd._modules[layer].bias = torch.nn.Parameter(bias, requires_grad=False)
            except:
                pass
        elif 'fc' in layer:
            weights = torch.cat([state_dict[i][f'{layer}.weight'] for i in range(args.votes)]).unsqueeze(dim=1)
            scd._modules[layer].weight = torch.nn.Parameter(weights, requires_grad=False)
            try:
                bias = torch.cat([state_dict[i][f'{layer}.bias'] for i in range(args.votes)])
                scd._modules[layer].bias = torch.nn.Parameter(bias, requires_grad=False)
            except:
                pass

    scd = ModelWrapper(scd)
    name = args.target if args.name == '' else args.name
    with open(f'checkpoints/{name}_{args.votes}.pkl', 'wb') as f:
        pickle.dump(scd, f)
        print(f'{name}_{args.votes}.pkl saved successfully')

    train_data, test_data, train_label, test_label = load_data(args.dataset, args.n_classes, c1=args.c0, c2=args.c1)
    if args.cnn:
        if 'cifar10' in args.dataset:
            train_data = train_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
            test_data = test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
        elif 'stl10' in args.dataset:
            train_data = train_data.reshape((-1, 96, 96, 3)).transpose((0, 3, 1, 2))
            test_data = test_data.reshape((-1, 96, 96, 3)).transpose((0, 3, 1, 2))
        elif 'celeba' in args.dataset:
            train_data = train_data.reshape((-1, 3, 96, 96))
            test_data = test_data.reshape((-1, 3, 96, 96))
        elif 'gtsrb' in args.dataset:
            train_data = train_data.reshape((-1, 3, 48, 48))
            test_data = test_data.reshape((-1, 3, 48, 48))
    if 'dm1' in args.target:
        train_data = train_data / 0.5 - 1
        test_data = test_data / 0.5 - 1

    start_time = time.time()
    yp = scd.predict(train_data, batch_size=200)
    end_time = time.time()
    train_acc = accuracy_score(y_true=train_label, y_pred=yp)
    print(f'{args.votes} votes train accuracy, '
          f'{train_acc} '
          f'inference cost %.2f seconds: ' % (end_time - start_time),
          )

    start_time = time.time()
    yp = scd.predict(test_data, batch_size=200)
    end_time = time.time()
    test_acc = accuracy_score(y_true=test_label, y_pred=yp)
    print(f'{args.votes} votes test accuracy: '
          f'{test_acc} '
          f'inference cost %.2f seconds: ' % (end_time - start_time),
          )

    save = True if args.save else False

    if save:
        path = os.path.join('logs', 'combined_acc')
        if not os.path.exists(path):
            try:
                os.makedirs(path)
            except:
                pass
        with open(os.path.join(path, f'{args.target}_{args.votes}'), 'w') as f:
            f.write('%.4f %.4f' % (train_acc, test_acc))
