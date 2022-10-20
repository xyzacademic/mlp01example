import torchvision
import numpy as np
import torchvision.transforms as transforms
import os
import torch
DATA = 'gtsrb'

if DATA == 'gtsrb':
    train_dir = '/home/y/yx277/research/ImageDataset/GTSRB/data/train_mc'
    test_dir = '/home/y/yx277/research/ImageDataset/GTSRB/data/test_mc'

test_transform = transforms.Compose(
                [
                transforms.Resize((48, 48)),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                # normalize,
                ])

train_dataset = torchvision.datasets.ImageFolder(root=train_dir, transform=test_transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=False,
                                           num_workers=16, pin_memory=True, drop_last=True)

test_dataset = torchvision.datasets.ImageFolder(root=test_dir, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False,
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

save_path = '../data/gtsrb'

if not os.path.isdir(save_path):
    os.mkdir(save_path)

os.chdir(save_path)
np.save('train_image.npy', train_data)
np.save('train_label.npy', train_labels)
np.save('test_image.npy', test_data)
np.save('test_label.npy', test_labels)
os.chdir(os.pardir)