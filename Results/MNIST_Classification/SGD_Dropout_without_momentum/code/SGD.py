import csv
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader

from data.dataset import MNISTDataset
from data.parser import parse_mnist
from data.transforms import MNISTTransform


class SGD_Hyper(object):

    def __init__(self, ):
        self.mode = 'dropout'  # 'dropout' or 'mlp'
        self.lr = 1e-3
        self.momentum = 0
        self.hidden_units = 400
        self.max_epoch = 600

        self.parse_seed = 0
        self.torch_seed = 0

        self.mnist_path = 'data/'
        self.num_valid = 10000
        self.batch_size = 125
        self.eval_batch_size = 1000
        self.num_workers = 1


class ModelMLPDropout(nn.Module):

    def __init__(self, hidden_units):
        super(ModelMLPDropout, self).__init__()
        self.fc0 = nn.Linear(784, hidden_units)
        self.fc1 = nn.Linear(hidden_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 10)

        torch.nn.init.kaiming_uniform_(self.fc0.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

    def forward(self, input_):
        input_ = F.dropout(input_, p=0.2, training=self.training)
        h1 = F.relu(self.fc0(input_))
        h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(self.fc1(h1))
        h2 = F.dropout(h2, p=0.5, training=self.training)
        h3 = self.fc2(h2)
        return h3


class ModelMLP(nn.Module):

    def __init__(self, hidden_units):
        super(ModelMLP, self).__init__()
        self.fc0 = nn.Linear(784, hidden_units)
        self.fc1 = nn.Linear(hidden_units, hidden_units)
        self.fc2 = nn.Linear(hidden_units, 10)

        torch.nn.init.kaiming_uniform_(self.fc0.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

    def forward(self, input_):
        h1 = F.relu(self.fc0(input_))
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
    torch.manual_seed(hyper.torch_seed)

    """Prepare data"""
    if train_loader is None or valid_loader is None or test_loader is None:
        train_data, train_label, valid_data, valid_label, test_data, test_label = parse_mnist(2, hyper.mnist_path,
                                                                                              hyper.num_valid, hyper.parse_seed)
        transform = MNISTTransform()  # divide by 126 as in paper
        train_dataset = MNISTDataset(train_data, train_label, transform=transform)
        valid_dataset = MNISTDataset(valid_data, valid_label, transform=transform)
        test_dataset = MNISTDataset(test_data, test_label, transform=transform)
        train_loader = DataLoader(train_dataset, hyper.batch_size, shuffle=True, num_workers=hyper.num_workers)
        valid_loader = DataLoader(valid_dataset, hyper.eval_batch_size, shuffle=False, num_workers=hyper.num_workers)
        test_loader = DataLoader(test_dataset, hyper.eval_batch_size, shuffle=False)

    """Set model"""
    if hyper.mode == 'mlp':
        model = ModelMLP(hyper.hidden_units)
    elif hyper.mode == 'dropout':
        model = ModelMLPDropout(hyper.hidden_units)
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
    path = 'Results/SGD_MNIST_' + hyper.mode + '_' + str(hyper.hidden_units) + '_' + str(hyper.lr)+ '_' + str(hyper.momentum)
    wr = csv.writer(open(path + '.csv', 'w'), delimiter=',', lineterminator='\n')
    wr.writerow(['epoch', 'valid_acc', 'test_acc', 'train_losses'])

    for i in range(hyper.max_epoch):
        wr.writerow((i + 1, 1 - valid_accs[i] / hyper.eval_batch_size,
                     1 - test_accs[i] / hyper.eval_batch_size, train_losses[i]))

    torch.save(model.state_dict(), path + '.pth')


if __name__ == '__main__':
    """ARGS"""
    if len(sys.argv) < 4:
        print('Call: python SGD.py [RATE 1-2] [UNITS] [EPOCHS]')
        sys.exit()

    hyper = SGD_Hyper()
    hyper.lr = 10 ** (- int(sys.argv[1]))
    hyper.hidden_units = int(sys.argv[2])  # in paper: 400, 800, 1200
    hyper.max_epoch = int(sys.argv[3])
    hyper.mode = sys.argv[4]  # either 'dropout' or 'mlp'

    print(hyper.lr, hyper.hidden_units, hyper.max_epoch, hyper.mode)
    SGD_run(hyper)
