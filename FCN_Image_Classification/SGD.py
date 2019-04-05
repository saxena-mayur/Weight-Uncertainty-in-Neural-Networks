import csv
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

torch.manual_seed(0)

class SGD_Hyper(object):

    def __init__(self, ):
        self.dataset = 'cifar10'  # mnist || cifar10 || fmnist

        self.mode = 'dropout'  # 'dropout' or 'mlp'
        self.lr = 1e-3
        self.momentum = 0.95
        self.hidden_units = 400
        self.max_epoch = 600

        self.mnist_path = 'data/'
        self.num_valid = 10000
        self.batch_size = 125
        self.eval_batch_size = 1000
        self.num_workers = 1

        self.n_input = 28*28
        self.n_ouput = 10


class ModelMLPDropout(nn.Module):

    def __init__(self, hidden_units, n_input=784, n_ouput=10):
        super(ModelMLPDropout, self).__init__()
        self.fc0 = nn.Linear(n_input, hidden_units)
        self.fc1 = nn.Linear(hidden_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, n_ouput)

        self.n_input = n_input

        torch.nn.init.kaiming_uniform_(self.fc0.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

    def forward(self, input_):
        input_ = F.dropout(input_.view(-1, self.n_input), p=0.2, training=self.training)
        h1 = F.relu(self.fc0(input_))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(self.fc1(h1))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h3 = self.fc2(h2)
        return h3


class ModelMLP(nn.Module):

    def __init__(self, hidden_units, n_input=784, n_ouput=10):
        super(ModelMLP, self).__init__()
        self.fc0 = nn.Linear(n_input, hidden_units)
        self.fc1 = nn.Linear(hidden_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, n_ouput)

        self.n_input = n_input

        torch.nn.init.kaiming_uniform_(self.fc0.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

    def forward(self, input_):
        h1 = F.relu(self.fc0(input_.view(-1, self.n_input)))
        h2 = F.relu(self.fc1(h1))
        h3 = self.fc2(h2)
        return h3


def train(model, optimizer, loader):
    model.train()
    loss_sum = 0
    acc_sum = 0
    for idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

        predict = output.data.max(1)[1]
        acc = predict.eq(target.data).cpu().sum().item()
        acc_sum += acc
    return loss_sum / len(loader), acc_sum / len(loader)


def evaluate(model, loader):
    model.eval()
    loss_sum = 0
    acc_sum = 0
    for idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss_sum += loss.item()

        predict = output.data.max(1)[1]
        acc = predict.eq(target.data).cpu().sum().item()
        acc_sum += acc
    return loss_sum / len(loader), acc_sum / len(loader)


def SGD_run(hyper, train_loader=None, valid_loader=None, test_loader=None):
    print(hyper.__dict__)

    """Prepare data"""
    if train_loader is None or valid_loader is None or test_loader is None:
        valid_size = 1 / 6

        if hyper.dataset == 'mnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255. / 126.),  # divide as in paper
            ])

            train_data = datasets.MNIST(
                root='data',
                train=True,
                download=False,
                transform=transform)
            test_data = datasets.MNIST(
                root='data',
                train=False,
                download=False,
                transform=transform)

            hyper.n_input = 28 * 28
            hyper.n_ouput = 10
        elif hyper.dataset == 'fmnist':
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255. / 126.),  # divide as in paper
            ])

            train_data = datasets.FashionMNIST(
                root='data2',
                train=True,
                download=False,
                transform=transform)
            test_data = datasets.FashionMNIST(
                root='data2',
                train=False,
                download=False,
                transform=transform)

            hyper.n_input = 28 * 28
            hyper.n_ouput = 10
        elif hyper.dataset == 'cifar10':
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),  # randomly flip and rotate
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

            train_data = datasets.CIFAR10(
                root='data',
                train=True,
                download=False,
                transform=transform)
            test_data = datasets.CIFAR10(
                root='data',
                train=False,
                download=False,
                transform=transform)

            hyper.n_input = 32 * 32 * 3  # 32x32 images of 3 colours
            hyper.n_ouput = 10
        else:
            raise ValueError('Unknown dataset')

        # obtain training indices that will be used for validation
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(valid_size * num_train)
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=hyper.batch_size,
            sampler=train_sampler,
            num_workers=1)
        valid_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=hyper.eval_batch_size,
            sampler=valid_sampler,
            num_workers=1)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=hyper.eval_batch_size,
            num_workers=1)

    """Set model"""
    if hyper.mode == 'mlp':
        model = ModelMLP(hyper.hidden_units, n_input=hyper.n_input, n_ouput=hyper.n_ouput)
    elif hyper.mode == 'dropout':
        model = ModelMLPDropout(hyper.hidden_units, n_input=hyper.n_input, n_ouput=hyper.n_ouput)
    else:
        raise ValueError('Not supported mode')

    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=hyper.lr, momentum=hyper.momentum)

    train_losses = np.zeros(hyper.max_epoch)
    valid_accs = np.zeros(hyper.max_epoch)
    test_accs = np.zeros(hyper.max_epoch)

    """Train"""
    for epoch in range(hyper.max_epoch):
        train_loss, train_acc = train(model, optimizer, train_loader)
        valid_loss, valid_acc = evaluate(model, valid_loader)
        test_loss, test_acc = evaluate(model, test_loader)

        print("{0:>5}: Error {1:.2f}%".format(
            epoch + 1, 100 * (1 - valid_acc / hyper.eval_batch_size)))

        print()

        valid_accs[epoch] = valid_acc
        test_accs[epoch] = test_acc
        train_losses[epoch] = train_loss

    """Save"""
    path = 'Results/SGD_' + hyper.dataset + '_' + hyper.mode + '_' + str(hyper.hidden_units) + '_' + str(hyper.lr)+ '_' + str(hyper.momentum)
    wr = csv.writer(open(path + '.csv', 'w'), delimiter=',', lineterminator='\n')
    wr.writerow(['epoch', 'valid_acc', 'test_acc', 'train_losses'])

    for i in range(hyper.max_epoch):
        wr.writerow((i + 1, 1 - valid_accs[i] / hyper.eval_batch_size,
                     1 - test_accs[i] / hyper.eval_batch_size, train_losses[i]))

    torch.save(model.state_dict(), path + '.pth')


if __name__ == '__main__':
    hyper = SGD_Hyper()    
    if len(sys.argv) > 1:
        hyper.hidden_units = int(sys.argv[1])

    if len(sys.argv) > 2:
        hyper.dataset = sys.argv[2]

    if len(sys.argv) > 3:
        hyper.mode = sys.argv[3]

    SGD_run(hyper)
