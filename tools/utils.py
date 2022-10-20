import time
import os
import pickle
import numpy as np
import sys
sys.path.append('..')
# from dataset import load_data
import torch
from torch.utils.data import TensorDataset, DataLoader
# import torchtext
import pandas as pd
from tqdm import tqdm

def save_checkpoint(obj, save_path, file_name, et, vc):
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    suffix = '%s'*6 % et[:6]
    new_name, extend = os.path.splitext(file_name)
    new_name = "%s_%s%s" % (new_name, suffix, extend)
    full_name = os.path.join(os.getcwd(), os.path.join(save_path, file_name))
    with open(full_name, 'wb') as f:
        pickle.dump(obj, f)
    print('Save %s successfully, verification code: %s' % (full_name, vc))


def print_title(vc_len=4):
    vc_table = [chr(i) for i in range(97, 123)]
    vc = ''.join(np.random.choice(vc_table, vc_len))
    print(' ')
    et = time.localtime()
    print('Experiment time: ', time.strftime("%Y-%m-%d %H:%M:%S", et))
    print('Verification code: ', vc)
    print('Args:')
    print(sys.argv)

    return et, vc


import torch

is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False
import torch.utils.data
import random


class BalancedBatchSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, labels=None):
        self.labels = labels
        self.dataset = dict()
        self.balanced_max = 0
        # Save all the indices for all the classes
        for idx in range(0, len(dataset)):
            label = self._get_label(dataset, idx)
            if label not in self.dataset:
                self.dataset[label] = list()
            self.dataset[label].append(idx)
            self.balanced_max = len(self.dataset[label]) \
                if len(self.dataset[label]) > self.balanced_max else self.balanced_max

        # Oversample the classes with fewer elements than the max
        for label in self.dataset:
            while len(self.dataset[label]) < self.balanced_max:
                self.dataset[label].append(random.choice(self.dataset[label]))
        self.keys = list(self.dataset.keys())
        self.currentkey = 0
        self.indices = [-1] * len(self.keys)

    def __iter__(self):
        while self.indices[self.currentkey] < self.balanced_max - 1:
            self.indices[self.currentkey] += 1
            yield self.dataset[self.keys[self.currentkey]][self.indices[self.currentkey]]
            self.currentkey = (self.currentkey + 1) % len(self.keys)
        self.indices = [-1] * len(self.keys)

    def _get_label(self, dataset, idx, labels=None):
        if self.labels is not None:
            return self.labels[idx].item()
        else:
            # Trying guessing
            dataset_type = type(dataset)
            if is_torchvision_installed and dataset_type is torchvision.datasets.MNIST:
                return dataset.train_labels[idx].item()
            elif is_torchvision_installed and dataset_type is torchvision.datasets.ImageFolder:
                return dataset.imgs[idx][1]
            else:
                raise Exception("You should pass the tensor of labels to the constructor as second argument")

    def __len__(self):
        return self.balanced_max * len(self.keys)


class MultiClassSampler(object):

    def __init__(self, dataset, classes=None, nrows=None, balanced=True,
                 transform=None):
        checkinstance(dataset, torch.utils.data.dataset.TensorDataset)
        self.dataset = dataset
        self.n_classes = classes
        self.keys = {}
        self.nrows = nrows
        self.balanced = balanced
        for i in range(self.n_classes):
            self.keys[i] = []
        for i in range(len(dataset)):
            self.keys[dataset[i][1].item()].append(i)
        self.indices_ratio = {}
        self.transform = transform

    def __iter__(self):
        if self.balanced:
            indices = []
            for i in range(self.n_classes):
                self.indices_ratio[i] = int(self.nrows * len(self.keys[i]))
                indices += np.random.choice(self.keys[i], self.indices_ratio[i], False).tolist()
        else:
            indices = np.random.choice(
                np.arange(self.__len__()), int(self.nrows * self.__len__()), False)
        return_set = self.dataset[indices]

        if self.transform is not None:
            return_set = (torch.stack([self.transform(return_set[0][i]) for i in range(len(indices))], dim=0), return_set[1])
            # return_set = (self.transform(return_set[0]), return_set[1])
        return return_set

    def __len__(self):
        return len(self.dataset)

    def next(self):

        return self.__iter__()

class MultiClassSampler2(object):

    def __init__(self, dataset, classes=None, nrows=None, fined_train_label=None, target_class=None):
        checkinstance(dataset, torch.utils.data.dataset.TensorDataset)
        self.fined_train_label = fined_train_label
        self.dataset = dataset
        self.n_classes = classes
        self.keys = {}
        self.onevall_classes = 2
        for i in range(self.n_classes):
            self.keys[i] = []
        for i in range(len(dataset)):
            self.keys[fined_train_label[i]].append(i)
        self.indices_ratio = {}
        for i in range(self.n_classes):
            self.indices_ratio[i] = int(nrows // 2 //(self.n_classes - 1))
        self.indices_ratio[target_class] = self.indices_ratio[0] * (self.n_classes - 1)

    def __iter__(self):
        indices = []
        for i in range(self.n_classes):
            indices += np.random.choice(self.keys[i], self.indices_ratio[i]).tolist()

        return self.dataset[indices]

    def __len__(self):
        return len(self.dataset)

    def next(self):

        return self.__iter__()


# class TextDataset(torchtext.data.Dataset):
#
#     def __init__(self, path, text_field, label_field, test=False, aug=False, **kwargs):
#         fields = [
#             ('text', text_field),
#             ('label', label_field),
#         ]
#
#         examples = []
#
#         csv_data = pd.read_csv(path)
#         print(f"read data from {path}")
#
#         if test:
#             for text in tqdm(csv_data['text']):
#                 examples.append(torchtext.data.Example.fromlist([text, None], fields))
#
#         else:
#             for text, label in tqdm(zip(csv_data['text'], csv_data['label'])):
#                 examples.append(torchtext.data.Example.fromlist([text, label], fields))
#         super(TextDataset, self).__init__(examples, fields, **kwargs)
#
#
def checkinstance(instance, target):
    if not isinstance(instance, target):
        raise TypeError(f'{instance} should be instance of class "{target}"'
                        f' but get {type(instance)}')



#
# if __name__ == '__main__':
#     train_data, test_data, train_label, test_label = load_data('cifar10', 10)
#     train_data = train_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
#     test_data = test_data.reshape((-1, 32, 32, 3)).transpose((0, 3, 1, 2))
#     np.random.seed(2018)
#
#     fined_train_label = train_label
#     y_train = np.zeros_like(train_label)
#     y_train[train_label == 4] = 1
#     y_test = np.zeros_like(test_label)
#     y_test[test_label == 4] = 1
#
#     train_label = y_train
#     test_label = y_test
#
#     trainset = TensorDataset(torch.from_numpy(train_data.astype(np.float32)),
#                              torch.from_numpy(train_label.astype(np.int64)))
#     testset = TensorDataset(torch.from_numpy(test_data.astype(np.float32)),
#                             torch.from_numpy(test_label.astype(np.int64)))
#
#     train_loader = MultiClassSampler2(dataset=trainset, classes=10,
#                                       nrows=1500, fined_train_label=fined_train_label, target_class=4)

