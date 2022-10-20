import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from advertorch.attacks import LinfPGDAttack, MomentumIterativeAttack, GradientSignAttack
import pickle
import pandas as pd
from tools import load_data

parser = argparse.ArgumentParser(description='PyTorch ImageNet Attack')
parser.add_argument('--input-dir', default='', help='Input directory with images.')
parser.add_argument('--output-dir', default='', help='Output directory with images.')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for adversarial attack')
parser.add_argument('--arch', default='resnet18', help='source model', )
parser.add_argument('--source', default='resnet50', help='source model', )
parser.add_argument('--target', default='resnet50', help='source model', )
parser.add_argument('--dataset', default='cifar10', help='dataset', )
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epsilon', default=8, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2, type=float,
                    help='perturb step size')
parser.add_argument('--gamma', default=0.2, type=float)
parser.add_argument('--momentum', default=1.0, type=float)
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--target_attack', type=int, default=0, metavar='N',
                    help='input batch size for adversarial attack')
parser.add_argument('--n_classes', type=int, default=10, metavar='N',
                    help='input batch size for adversarial attack')
parser.add_argument('--attack_type', default='pgd', type=str,
                    help='attack method')
parser.add_argument('--name', default='', type=str, help='output csv file')
parser.add_argument('--c0', default=0, type=int,
                    help='perturb number of steps')
parser.add_argument('--c1', default=0, type=int,
                    help='perturb number of steps')
args = parser.parse_args()
use_cuda = True


def evaluation(data_loader, use_cuda, net, correct_index=None):
    a = time.time()
    net.eval()
    correct = 0
    yp = []
    label = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # data = data.transpose(1, 3).reshape((data.size(0), -1))
            outputs = net(data).flatten()
            # print(outputs.shape)
            outputs = outputs.round()
            yp.append(outputs)
            label.append(target)
            correct += outputs.eq(target).sum().item()

        yp = torch.cat(yp, dim=0)
        label = torch.cat(label, dim=0)
        acc = (yp == label).float()
        if correct_index is not None:
            acc = acc[correct_index].mean().item()
        else:
            acc = acc.mean().item()
        print("Accuracy: {:.5f}, "
              "cost {:.2f} seconds".format(acc, time.time() - a))

    return yp.cpu(), (yp == label).cpu(), acc


def evaluation_cnn01(data_loader, use_cuda, net, correct_index=None):
    a = time.time()
    net.eval()
    correct = 0
    yp = []
    label = []
    for batch_idx, (data, target) in enumerate(data_loader):
        if use_cuda:
            data, target = data.cuda(), target.cuda()
        # data = data.type_as(net._modules[list(net._modules.keys())[0]].weight)

        outputs = net(data).flatten()
        # print(outputs.shape)
        yp.append(outputs)
        label.append(target)

    yp = torch.cat(yp, dim=0)
    label = torch.cat(label, dim=0)
    acc = (yp == label).float()
    if correct_index is not None:
        acc = acc[correct_index].mean().item()
    else:
        acc = acc.mean().item()
    print("Accuracy: {:.5f}, "
          "cost {:.2f} seconds".format(acc, time.time() - a))

    return yp.cpu(), (yp == label).cpu()


def generate_adversarial_example(model, data_loader, adversary):
    """
    generate and save adversarial example
    """
    model.eval()
    advs = []
    labels = []
    for batch_idx, (inputs, target) in enumerate(data_loader):
        if use_cuda:
            inputs = inputs.cuda()
            target = target.cuda()
        # with torch.no_grad():
        #     # _, pred = model(inputs).topk(1, 1, True, True)
        #     pred = model(inputs).argmax(dim=1)

        # craft adversarial images
        if args.target_attack:
            target = (target + 1) % args.n_classes
        inputs_adv = adversary.perturb(inputs, target.float())
        advs.append(inputs_adv)
        labels.append(target)
        # save adversarial images
    return torch.cat(advs, dim=0).cpu(), torch.cat(labels, dim=0).cpu()


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465])
        self.std = torch.tensor([0.247, 0.243, 0.261])

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :,
                                                                 None, None]


class Dm(nn.Module):
    def __init__(self):
        super(Dm, self).__init__()

    def forward(self, x):
        return x / 0.5 - 1


if __name__ == '__main__':
    # train_data, test_data, train_label, test_label = load_data('cifar10', 10)
    # train_data = train_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
    # test_data = test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
    # # test_data = np.random.normal(size=(2000, 3, 224, 224)).astype(np.float32)
    # trainset = TensorDataset(torch.from_numpy(train_data.astype(np.float32)),
    #                          torch.from_numpy(train_label.astype(np.int64)))
    # testset = TensorDataset(torch.from_numpy(test_data.astype(np.float32)),
    #                         torch.from_numpy(test_label.astype(np.int64)))
    #
    # test_loader = DataLoader(testset, batch_size=16, shuffle=False, num_workers=0,
    #                          pin_memory=False)
    # DATA = 'cifar10'
    #
    # if DATA == 'cifar10':
    #     train_dir = '/home/y/yx277/research/ImageDataset/cifar10'
    #     test_dir = '/home/y/yx277/research/ImageDataset/cifar10'
    #
    # test_transform_list = [transforms.ToTensor()]
    # test_transform = transforms.Compose(test_transform_list)
    # testset = torchvision.datasets.CIFAR10(root=test_dir, train=False, download=True,
    #                                        transform=test_transform)
    #
    # test_idx = [True if i < args.n_classes else False for i in testset.targets]
    # testset.data = testset.data[test_idx]
    #
    # testset.targets = [i for i in testset.targets if i < args.n_classes]

    train_data, test_data, train_label, test_label = load_data(args.dataset, args.n_classes, c1=args.c0, c2=args.c1)

    testset = TensorDataset(torch.from_numpy(test_data.astype(np.float32)),
                            torch.from_numpy(test_label.astype(np.int64)))

    test_loader = DataLoader(testset, batch_size=2000,
                             shuffle=False, num_workers=0,
                             pin_memory=False)

    source_list = [
        'mlp01scd',
    ]

    target_list = [
        'mlp01scd',
    ]

    df = pd.DataFrame(columns=['model'] + target_list)
    df['model'] = source_list

    for i, source_model in enumerate(source_list):
        model_name = f'{args.dataset}_{args.c0}{args.c1}_{source_model}_8'
        with open(f'../main/checkpoints/{model_name}.pkl', 'rb') as f:
            scd = pickle.load(f)
            model = scd.net
            orig_act = model.signb
            # model.signb = torch.sigmoid
            model.float()
        print(f'gamma: {args.gamma}')
        print(f'arch: {args.arch}')
        print(f'epsilon: {args.epsilon}')
        # model.load_state_dict(torch.load(os.path.join('checkpoints', source_model))['net'])
        if 'normalize1' in source_model:
            model = nn.Sequential(Normalize(), model)
        if 'dm1' in source_model:
            model = nn.Sequential(Dm(), model)
        # model = pretrainedmodels.__dict__[args.arch](num_classes=1000, pretrained='imagenet')
        if use_cuda:
            model.cuda()
        print(f'Source model ("{source_model}") on clean data: ')
        yp, correct_index, acc = evaluation(test_loader, use_cuda, model)

        epsilon = args.epsilon / 255.0
        if args.step_size < 0:
            step_size = epsilon / args.num_steps
        else:
            step_size = args.step_size / 255.0

        if args.attack_type == 'fgsm':
            print('using FGSM attack'.format(args.momentum))
            adversary = GradientSignAttack(predict=lambda x: model(x).flatten(),
                                           loss_fn=nn.BCELoss(reduction='mean'),
                                           eps=epsilon,
                                           clip_min=0.0, clip_max=1.0,
                                           targeted=bool(args.target_attack))
        elif args.attack_type == 'pgd':
            print('using linf PGD attack')
            adversary = LinfPGDAttack(
                # predict=model,
                # loss_fn=nn.CrossEntropyLoss(reduction="mean"),
                predict=lambda x: model(x).flatten(),
                # loss_fn=nn.NLLLoss(reduction="mean"),
                loss_fn=nn.BCELoss(reduction='mean'),
                eps=epsilon, nb_iter=args.num_steps, eps_iter=step_size,
                rand_init=False, clip_min=0.0, clip_max=1.0, targeted=bool(args.target_attack))

        elif args.attack_type == 'mia':
            print('using linf momentum iterative attack')
            adversary = MomentumIterativeAttack(
                # predict=model,
                # loss_fn=nn.CrossEntropyLoss(reduction="mean"),
                predict=lambda x: model(x).flatten(),
                # loss_fn=nn.NLLLoss(reduction="mean"),
                loss_fn=nn.BCELoss(reduction='mean'),
                eps=epsilon, nb_iter=args.num_steps, eps_iter=step_size,
                decay_factor=args.momentum, clip_min=0.0, clip_max=1.0,
                targeted=bool(args.target_attack))

        model.signb = torch.sigmoid
        # print('scaling weights')
        # model._modules['fc2_si'].weight /= 100
        # model._modules['fc2_si'].bias /= 100
        # print(f'Source model ("{source_model}") on clean data: ')
        # yp, correct_index, acc = evaluation(test_loader, use_cuda, model)
        test_adv, source_adv_target = generate_adversarial_example(model=model,
                                                                   data_loader=test_loader,
                                                                   adversary=adversary)

        # print('recovery weights')
        # model._modules['fc2_si'].weight *= 100
        # model._modules['fc2_si'].bias *= 100

        model.signb = orig_act
        # if args.target_attack:
        #     np.save('toy3_ensemble_ta_adv.npy', test_adv.cpu().numpy())
        #     np.save('toy3_ensemble_ta_target.npy', source_adv_target.cpu().numpy())
        # else:
        #     np.save('toy3_ensemble_ut_adv.npy', test_adv.cpu().numpy())
        #     np.save('toy3_ensemble_ut_target.npy', source_adv_target.cpu().numpy())

        # if args.target_attack:
        #     test_adv = torch.from_numpy(np.load('toy3_ensemble_ta_adv.npy')).cuda()
        #     source_adv_target = torch.from_numpy(np.load('toy3_ensemble_ta_target.npy'))
        # else:
        #     test_adv = torch.from_numpy(np.load('toy3_ensemble_ut_adv.npy')).cuda()
        #     source_adv_target = torch.from_numpy(np.load('toy3_ensemble_ut_target.npy'))
        try:
            advset = TensorDataset(test_adv, torch.from_numpy(test_label).long())
        except:
            advset = TensorDataset(test_adv, torch.LongTensor(testset.labels))
        adv_loader = DataLoader(advset, batch_size=args.batch_size, shuffle=False, num_workers=4,
                                pin_memory=False)
        k = testset.tensors[0] - advset.tensors[0]
        print('l2 distance: ', torch.norm(k, dim=1).mean())
        if args.target_attack:
            advset_target = TensorDataset(test_adv, source_adv_target)
            adv_target_loader = DataLoader(advset_target, batch_size=args.batch_size, shuffle=False,
                                           num_workers=4,
                                           pin_memory=False)
            print(f'Source model ("{source_model}") match rate on adv target data (all):')
            _, _, acc = evaluation(adv_target_loader, use_cuda, model)
            # print(f'Source model ("{source_model}") match rate on adv target data (correct):')
            # evaluation(adv_target_loader, use_cuda, model, correct_index)
        else:
            print(f'Source model ("{source_model}") on adv data (all):')
            source_adv_yp, _, acc = evaluation(adv_loader, use_cuda, model)
            # print(f'Source model ("{source_model}") on adv data (correct):')
            # source_adv_yp, _ = evaluation(adv_loader, use_cuda, model, correct_index)

        for target_model in target_list:
            target_model_name = f'{args.dataset}_{args.c0}{args.c1}_{target_model}_8'
            try:
                with open(os.path.join('../experiments/checkpoints', f'{target_model_name}.pkl'),
                          'rb') as f:
                    scd = pickle.load(f).net
                    scd.float()

                # scd.load_state_dict(torch.load(os.path.join('checkpoints', target_model))['net'])
                if 'normalize1' in target_model:
                    scd = nn.Sequential(Normalize(), scd)
                if 'dm1' in target_model:
                    scd = nn.Sequential(Dm(), scd)
                if use_cuda:
                    scd.cuda()
                print(f'Target model ("{target_model}") on clean data:')

                yp, correct_index, clean_acc = evaluation(test_loader, use_cuda, scd)

                # print(f'Source model ("{source_model}") on adv data:')
                # evaluation(adv_loader, use_cuda, model)
                if args.target_attack:
                    # try:
                    #     print(f'Target model ("{target_model}") match rate on adv target data (all):')
                    #     evaluation_cnn01(adv_target_loader, use_cuda, scd)
                    #     print(
                    #         f'Target model ("{target_model}") match rate on adv target data (correct):')
                    #     evaluation_cnn01(adv_target_loader, use_cuda, scd, correct_index)
                    # except:
                    print(f'Target model ("{target_model}") match rate on adv target data (all):')
                    _, _, adv_acc = evaluation(adv_target_loader, use_cuda, scd)
                    # print(
                    #     f'Target model ("{target_model}") match rate on adv target data (correct):')
                    # evaluation(adv_target_loader, use_cuda, scd, correct_index)
                else:
                    # try:
                    #     print(f'Target model ("{target_model}") on adv data (all):')
                    #     evaluation_cnn01(adv_loader, use_cuda, scd)
                    #     print(f'Target model ("{target_model}") on adv data (correctly):')
                    #     evaluation_cnn01(adv_loader, use_cuda, scd, correct_index)
                    # except:
                    print(f'Target model ("{target_model}") on adv data (all):')
                    _, _, adv_acc = evaluation(adv_loader, use_cuda, scd)
                    # print(f'Target model ("{target_model}") on adv data (correctly):')
                    # evaluation(adv_loader, use_cuda, scd, correct_index)
                # df.at[i, target_model] = '%.4f (%.4f)' % (adv_acc, clean_acc)
                df.at[i, target_model] = '%.4f' % (adv_acc)
                # df.at[i, target_model] = '%.4f' % (adv_acc*100)
            except:
                continue

    if args.name != '':
        df.to_csv(f'{args.name}.csv', index=False)
