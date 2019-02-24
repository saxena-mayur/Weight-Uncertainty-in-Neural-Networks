# https://gist.github.com/Fuchai/12f2321e6c8fa53058f5eb23aeddb6ab
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np


class GenHelper(Dataset):
    def __init__(self, mother, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping = mapping
        self.length = length
        self.mother = mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length


def train_valid_split(ds, split_fold=10):
    '''
    This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
    efficiently.
    :return:
    '''

    dslen = len(ds)
    indices = list(range(dslen))
    valid_size = dslen//split_fold
    np.random.shuffle(indices)
    train_mapping = indices[valid_size:]
    valid_mapping = indices[:valid_size]
    train = GenHelper(ds, dslen - valid_size, train_mapping)
    valid = GenHelper(ds, valid_size, valid_mapping)

    return train, valid


def load_mnist(batch_size):

    T = transforms.Compose([
        transforms.ToTensor(),  # divides by 255
        # correct division by 126 as in paper
        transforms.Lambda(lambda x: (x * 255.) / 126.)
        # transforms.Normalize((0.1307,), (0.3081,)) # global mean and stdv
    ])

    train_set = dset.MNIST(root='mnist', train=True,
                           transform=T, download=True)
    test_set = dset.MNIST(root='mnist', train=False,
                          transform=T, download=True)
    train_set, valid_set = train_valid_split(train_set, split_fold=6)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True)  # 50000 images

    valid_loader = torch.utils.data.DataLoader(
        dataset=valid_set,
        batch_size=batch_size,
        shuffle=True)  # 10000 images

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False)  # 10000 images

    # examples = enumerate(test_loader)
    # batch_idx, (example_data, example_targets) = next(examples)
    # print(example_data[0][0])

    return train_loader, valid_loader, test_loader
