import os
import numpy as np
from sklearn.preprocessing import normalize, Normalizer
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torchvision import transforms


def get_data(data=None):
    if data == 'mnist':
        train_dir = '../data'
        test_dir = '../data'

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ])

        trainset = torchvision.datasets.MNIST(root=train_dir, train=True, download=True, transform=test_transform)
        testset = torchvision.datasets.MNIST(root=test_dir, train=False, download=True, transform=test_transform)

        save_path = '../data/mnist'

        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        train_data = trainset.data.numpy()
        train_label = np.array(trainset.targets)
        test_data = testset.data.numpy()
        test_label = np.array(testset.targets)

        os.chdir(save_path)
        np.save('train_image.npy', train_data)
        np.save('train_label.npy', train_label)
        np.save('test_image.npy', test_data)
        np.save('test_label.npy', test_label)
        os.chdir(os.pardir)


    elif data == 'cifar10':
        curdir = os.getcwd()
        train_dir = '../data'
        test_dir = '../data'

        test_transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ])

        trainset = torchvision.datasets.CIFAR10(root=train_dir, train=True, download=True, transform=test_transform)

        testset = torchvision.datasets.CIFAR10(root=test_dir, train=False, download=True, transform=test_transform)

        save_path = '../data/cifar10'

        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        os.chdir(save_path)
        np.save('train_image.npy', trainset.data)
        np.save('train_label.npy', np.array(trainset.targets))
        np.save('test_image.npy', testset.data)
        np.save('test_label.npy', np.array(testset.targets))
        os.chdir(curdir)

    elif data == 'stl10':
        pass

    elif data == 'imagenet':
        train_dir = '../data/sub_imagenet/train_mc'
        test_dir = '../data/sub_imagenet/val_mc'

        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize,
            ])

        train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=test_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False,
                                                   num_workers=16, pin_memory=True, drop_last=True)

        test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False,
                                                  num_workers=10, pin_memory=True)

        train_data = []
        test_data = []
        train_labels = []
        test_labels = []

        for idx, (data, targets) in enumerate(train_loader):
            print(idx)
            train_data.append(data.numpy())
            train_labels.append(targets.numpy())

        train_data = np.concatenate(train_data, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)

        for idx, (data, targets) in enumerate(test_loader):
            print(idx)
            test_data.append(data.numpy())
            test_labels.append(targets.numpy())

        test_data = np.concatenate(test_data, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)

        save_path = '../data/imagenet'

        if not os.path.isdir(save_path):
            os.makedirs(save_path, exist_ok=True)

        os.chdir(save_path)
        np.save('train_image.npy', train_data)
        np.save('train_label.npy', train_labels)
        np.save('test_image.npy', test_data)
        np.save('test_label.npy', test_labels)
        os.chdir(os.pardir)


def load_data(data=None, n_classes=2, c1=None, c2=None):
    if data == 'mnist':
        curdir = os.getcwd()
        os.chdir('../data/mnist')
        train_data = np.load('train_image.npy').reshape((-1, 28 * 28)) / 255
        test_data = np.load('test_image.npy').reshape((-1, 28 * 28)) / 255
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)


    elif data == 'ptbdb':
        curdir = os.getcwd()
        os.chdir('../data/ptbdb')
        normal = pd.read_csv('ptbdb_normal.csv', header=None)
        abnormal = pd.read_csv('ptbdb_abnormal.csv', header=None)
        raw_data = normal.append(abnormal).loc[:, :186].values.reshape(-1, 187)  # .fillna(method='ffill')
        labels = normal.append(abnormal).loc[:, 187].values.flatten()
        # ohe = OneHotEncoder()
        # labels = ohe.fit_transform(labels).toarray()
        train_data, test_data, train_label, test_label = train_test_split(raw_data, labels, train_size=0.9,
                                                                          random_state=5947)
        os.chdir(curdir)

    elif data == 'cifar10':
        curdir = os.getcwd()
        path = 'data/cifar10'

        if not os.path.exists(path) or not os.path.exists(
                os.path.join(path, 'train_image.npy')
        ):
            get_data('cifar10')

        os.chdir(path)
        train_data = np.load('train_image.npy').reshape((-1, 32 * 32 * 3)) / 255
        test_data = np.load('test_image.npy').reshape((-1, 32 * 32 * 3)) / 255
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)

    elif data == 'cifar10_simclr':
        curdir = os.getcwd()
        os.chdir('../data/cifar10_simclr')
        train_data = np.load('train_image.npy')
        test_data = np.load('test_image.npy')
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)


    elif data == 'cifar10_binary':
        curdir = os.getcwd()
        path = '../data/cifar10'
        if not os.path.exists(path) or not os.path.exists(
                os.path.join(path, 'train_image.npy')
        ):
            get_data('cifar10')

        os.chdir(path)
        train_data = np.load('train_image.npy').reshape((-1, 32 * 32 * 3)) / 255
        test_data = np.load('test_image.npy').reshape((-1, 32 * 32 * 3)) / 255
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')

        t1 = c1
        t2 = c2

        train_data_a = train_data[train_label == t1]
        train_data_b = train_data[train_label == t2]
        train_label_a = np.zeros((train_data_a.shape[0],), dtype=np.int64)
        train_label_b = np.ones((train_data_b.shape[0],), dtype=np.int64)

        train_data = np.concatenate([train_data_a, train_data_b], axis=0)
        train_label = np.concatenate([train_label_a, train_label_b], axis=0)

        test_data_a = test_data[test_label == t1]
        test_data_b = test_data[test_label == t2]
        test_label_a = np.zeros((test_data_a.shape[0],), dtype=np.int64)
        test_label_b = np.ones((test_data_b.shape[0],), dtype=np.int64)

        test_data = np.concatenate([test_data_a, test_data_b], axis=0)
        test_label = np.concatenate([test_label_a, test_label_b], axis=0)

        os.chdir(curdir)

    elif data == 'cifar10_rdcnn':
        curdir = os.getcwd()
        os.chdir('../data/cifar10')
        train_data = np.load('train_data_3_1_1000.npy')
        test_data = np.load('test_data_3_1_1000.npy')
        train_label = np.load('train_label_3_1_1000.npy')
        test_label = np.load('test_label_3_1_1000.npy')

        # ne = Normalizer()
        # train_data = normalize(train_data, axis=0)
        # test_data = normalize(test_data, axis=0)
        os.chdir(curdir)

    elif data == 'stl10':
        curdir = os.getcwd()
        os.chdir('../data/stl10')
        train_data = np.load('train_image.npy').reshape((-1, 96 * 96 * 3)) / 255
        test_data = np.load('test_image.npy').reshape((-1, 96 * 96 * 3)) / 255
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)

    elif data == 'imagenet':
        curdir = os.getcwd()
        os.chdir('../data/imagenet')
        train_data = np.load('train_image.npy').reshape((-1, 224 * 224 * 3))
        test_data = np.load('test_image.npy').reshape((-1, 224 * 224 * 3))
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)

    elif data == 'gtsrb':
        curdir = os.getcwd()
        os.chdir('../data/gtsrb')
        train_data = np.load('train_image.npy').reshape((-1, 48 * 48 * 3))
        test_data = np.load('test_image.npy').reshape((-1, 48 * 48 * 3))
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)

    elif data == 'gtsrb_binary':
        curdir = os.getcwd()
        os.chdir('../data/gtsrb_binary')
        train_data = np.load('train_image.npy').reshape((-1, 48 * 48 * 3))
        test_data = np.load('test_image.npy').reshape((-1, 48 * 48 * 3))
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)

    elif data == 'celeba':
        curdir = os.getcwd()
        os.chdir('../data/celeba')
        train_data = np.load('train_image.npy').reshape((-1, 96 * 96 * 3))
        test_data = np.load('test_image.npy').reshape((-1, 96 * 96 * 3))
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)

    elif data == 'ecg':
        curdir = os.getcwd()
        os.chdir('../data/ecg')
        train_data = np.load('train_image.npy')
        test_data = np.load('test_image.npy')
        mean = train_data.mean()
        std = train_data.std()
        print('mean: ', mean)
        print('std: ', std)
        train_data = (train_data - mean) / std
        test_data = (test_data - mean) / std
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)

    elif data == 'fake':
        curdir = os.getcwd()
        os.chdir('../data/fake')
        train_data = np.load('train_image.npy')
        test_data = np.load('test_image.npy')
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)

    elif data == 'imdb':
        curdir = os.getcwd()
        os.chdir('../data/imdb')
        train_data = np.load('train_image.npy')
        test_data = np.load('test_image.npy')
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)

    elif data == 'yelp':
        curdir = os.getcwd()
        os.chdir('../data/yelp')
        train_data = np.load('train_image.npy')
        test_data = np.load('test_image.npy')
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)


    elif data == 'mr':
        curdir = os.getcwd()
        os.chdir('../data/mr')
        train_data = np.load('train_image.npy')
        test_data = np.load('test_image.npy')
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)


    elif data == 'ag':
        curdir = os.getcwd()
        os.chdir('../data/ag')
        train_data = np.load('train_image.npy')
        test_data = np.load('test_image.npy')
        train_label = np.load('train_label.npy')
        test_label = np.load('test_label.npy')
        os.chdir(curdir)

    elif data == 'toy':
        curdir = os.getcwd()
        train_data = np.array([[1, 1],
                               [1, 2],
                               [3, 1],
                               [3, 2],
                               [5, 1]], dtype=np.float32)
        train_label = np.array([0, 0, 1, 1, 0], dtype=np.int8)
        test_data = train_data
        test_label = train_label
        os.chdir(curdir)

    elif data == 'random':
        curdir = os.getcwd()
        train_data = np.random.normal(size=(248, 3, 150, 150))
        test_data = np.random.normal(size=(128, 3, 150, 150))
        train_label = np.random.randint(0, 2, size=(248,)).astype(np.int8)
        test_label = np.random.randint(0, 2, size=(128,)).astype(np.int8)
        os.chdir(curdir)

    else:
        raise AssertionError("%s is not in the list" % data)

    train = train_data[train_label < n_classes]
    test = test_data[test_label < n_classes]
    train_label = train_label[train_label < n_classes]
    test_label = test_label[test_label < n_classes]

    return train, test, train_label, test_label


if __name__ == '__main__':
    train, test, train_label, test_label = load_data('mnist')