# Credit: Kyuhong Shim(skhu20@snu.ac.kr) - https://github.com/khshim/pytorch_mnist
import sys
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

from data.parser import parse_mnist
from data.dataset import MNISTDataset
from data.transforms import MNISTTransform


class Config(object):

    def __init__(self):

        self.mode = 'dropout'  # 'dropout' or 'mlp'
        self.parse_seed = 1
        self.torch_seed = 1

        self.mnist_path = 'data/'
        self.num_valid = 10000
        self.batch_size = 128
        self.eval_batch_size = 1000
        self.num_workers = 4


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


def main(cfg, lr, hidden_units, max_epoch):
    torch.manual_seed(cfg.torch_seed)

    """Prepare data"""
    if cfg.mode == 'mlp' or cfg.mode == 'dropout':
        train_data, train_label, valid_data, valid_label, test_data, test_label = parse_mnist(
            2, cfg.mnist_path, cfg.num_valid, cfg.parse_seed)
    # elif cfg.mode == 'cnn':
    #     train_data, train_label, valid_data, valid_label, test_data, test_label = parse_mnist(4, cfg.mnist_path, cfg.num_valid, cfg.parse_seed)
    else:
        raise ValueError('Not supported mode')

    transform = MNISTTransform()
    train_dataset = MNISTDataset(train_data, train_label, transform=transform)
    valid_dataset = MNISTDataset(valid_data, valid_label, transform=transform)
    test_dataset = MNISTDataset(test_data, test_label, transform=transform)
    train_iter = DataLoader(train_dataset, cfg.batch_size,
                            shuffle=True, num_workers=cfg.num_workers)
    valid_iter = DataLoader(
        valid_dataset, cfg.eval_batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_iter = DataLoader(test_dataset, cfg.eval_batch_size, shuffle=False)

    """Set model"""
    if cfg.mode == 'mlp':
        model = ModelMLP(hidden_units)
    elif cfg.mode == 'dropout':
        model = ModelMLPDropout(hidden_units)
    else:
        raise ValueError('Not supported mode')

    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_losses = np.zeros(max_epoch)
    valid_losses = np.zeros(max_epoch)
    test_losses = np.zeros(max_epoch)
    valid_accs = np.zeros(max_epoch)
    test_accs = np.zeros(max_epoch)

    """Train"""
    for epoch in range(max_epoch):
        train_loss, train_acc = train(model, optimizer, train_iter)
        valid_loss, valid_acc = evaluate(model, valid_iter)
        test_loss, test_acc = evaluate(model, test_iter)

        print("{0:>5}: Error {1:.2f}%".format(
            epoch + 1, 100 * (1 - valid_acc / cfg.eval_batch_size)))

        train_losses[epoch] = train_loss
        valid_losses[epoch] = valid_loss
        test_losses[epoch] = test_loss
        valid_accs[epoch] = valid_acc
        test_accs[epoch] = test_acc

    """Save"""
    suffix = str(lr) + '_' + str(hidden_units)

    wr = csv.writer(open('results/results_' + suffix + '.csv', 'w'), delimiter=',', lineterminator='\n')
    wr.writerow(['epoch', 'train_losses', 'valid_losses',
                 'valid_correct', 'test_losses', 'test_correct'])
    for i in range(max_epoch):
        wr.writerow((i + 1, train_losses[i], valid_losses[i],
                     1 - valid_accs[i] / cfg.eval_batch_size, test_losses[i], 1 - test_accs[i] / cfg.eval_batch_size))

    torch.save(model.state_dict(), 'results/model_' + suffix + '.pth')


if __name__ == '__main__':
    """ARGS"""
    if len(sys.argv) < 4:
        print('Call: python SGD.py [RATE 1-4] [UNITS 1-3] [EPOCHS]')
        sys.exit()

    lrs = [1e-3, 1e-4, 1e-5, 1e-2]
    lr = lrs[int(sys.argv[1]) - 1]
    hidden_units = 400 * int(sys.argv[2])  # in paper: 400, 800, 1200
    enable_dropout = True
    max_epoch = int(sys.argv[3])

    config = Config()
    main(config, lr, hidden_units, max_epoch)
