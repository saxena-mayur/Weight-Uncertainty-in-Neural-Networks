import csv
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets

torch.manual_seed(0)


class BBB_Hyper(object):

    def __init__(self, ):
        self.dataset = 'mnist'  # 'mnist' or 'cifar10'

        # ADAPT THE FOLLOWING TWO PARAMS TO YOUR PROBLEM
        self.multiplier = 10 ** 4.5 # 4.5 is working fine for classification, but may need to be adapted
        self.lr = 1e-6 # 1e-3 | 1e-4 | 1e-5 | 1e-6

        self.momentum = 0.95
        self.hidden_units = 1200
        self.s1 = float(np.exp(-6))
        self.s2 = float(np.exp(-1))
        self.pi = 0.75

        self.mixture = False # better results for False, but more stable for True
        self.google_init = False # can also be False, then valid and test closer together

        self.max_epoch = 600
        self.n_samples = 1
        self.batch_size = 125
        self.eval_batch_size = 1000


GAUSSIAN_SCALER = 1. / np.sqrt(2.0 * np.pi)
def gaussian(x, mu, sigma):
    bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    return GAUSSIAN_SCALER / sigma * bell


def mixture_prior(input, pi, s1, s2):
    prob1 = pi * gaussian(input, 0., s1)
    prob2 = (1. - pi) * gaussian(input, 0., s2)
    return torch.log(prob1 + prob2)


def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu) ** 2 / (2 * sigma ** 2)


def log_gaussian_rho(x, mu, rho):
    return float(-0.5 * np.log(2 * np.pi)) - rho - (x - mu) ** 2 / (2 * torch.exp(rho) ** 2)


class BBBLayer(nn.Module):
    def __init__(self, n_input, n_output, hyper):
        super(BBBLayer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output

        self.s1 = hyper.s1
        self.s2 = hyper.s2
        self.pi = hyper.pi

        if hyper.google_init:
            self.weight_mu = nn.Parameter(torch.Tensor(n_output, n_input).normal_(0., .05))  # or .01
            self.weight_rho = nn.Parameter(torch.Tensor(n_output, n_input).normal_(-5., .05))  # or -4
            self.bias_mu = nn.Parameter(torch.Tensor(n_output).normal_(0., .05))
            self.bias_rho = nn.Parameter(torch.Tensor(n_output).normal_(-5., .05))

        else:
            self.weight_mu = nn.Parameter(torch.Tensor(n_output, n_input).normal_(0, 0.01))
            self.weight_rho = nn.Parameter(torch.Tensor(n_output, n_input).normal_(0, 0.01))
            self.bias_mu = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
            self.bias_rho = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))

        self.lpw = 0.
        self.lqw = 0.

    def forward(self, data, infer=False):
        if infer:
            output = F.linear(data, self.weight_mu, self.bias_mu)
            return output

        epsilon_W = Variable(torch.Tensor(self.n_output, self.n_input).normal_(0, self.s1).cuda())
        epsilon_b = Variable(torch.Tensor(self.n_output).normal_(0, self.s1).cuda())
        W = self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * epsilon_W
        b = self.bias_mu + torch.log1p(torch.exp(self.bias_rho)) * epsilon_b

        output = F.linear(data, W, b)

        self.lqw = log_gaussian_rho(W, self.weight_mu, self.weight_rho).sum() + \
                   log_gaussian_rho(b, self.bias_mu, self.bias_rho).sum()
        if hyper.mixture:
            self.lpw = mixture_prior(W, self.pi, self.s2, self.s1).sum() + \
                       mixture_prior(b, self.pi, self.s2, self.s1).sum()
        else:
            self.lpw = log_gaussian(W, 0, self.s1).sum() + log_gaussian(b, 0, self.s1).sum()

        return output


class BBB(nn.Module):
    def __init__(self, n_input, n_ouput, hyper):
        super(BBB, self).__init__()

        self.n_input = n_input
        self.layers = nn.ModuleList([])
        self.layers.append(BBBLayer(n_input, hyper.hidden_units, hyper))
        self.layers.append(BBBLayer(hyper.hidden_units, hyper.hidden_units, hyper))
        self.layers.append(BBBLayer(hyper.hidden_units, n_ouput, hyper))

    def forward(self, data, infer=False):
        output = F.relu(self.layers[0](data.view(-1, self.n_input), infer))
        output = F.relu(self.layers[1](output, infer))
        output = F.log_softmax(self.layers[2](output, infer), dim=1)
        return output

    def get_lpw_lqw(self):
        lpw = self.layers[0].lpw + self.layers[1].lpw + self.layers[2].lpw
        lqw = self.layers[0].lqw + self.layers[1].lqw + self.layers[2].lqw
        return lpw, lqw


def probs(model, hyper, data, target):
    s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
    for _ in range(hyper.n_samples):
        output = model(data)

        sample_log_pw, sample_log_qw = model.get_lpw_lqw()
        sample_log_likelihood = -F.nll_loss(output, target, reduction='sum') * hyper.multiplier

        s_log_pw += sample_log_pw / hyper.n_samples
        s_log_qw += sample_log_qw / hyper.n_samples
        s_log_likelihood += sample_log_likelihood / hyper.n_samples

    return s_log_pw, s_log_qw, s_log_likelihood

def ELBO(hyper, l_pw, l_qw, l_likelihood, n_batches):
    return ((1. / n_batches) * (l_qw - l_pw) - l_likelihood) / hyper.batch_size


def train(model, optimizer, loader, train=True):
    loss_sum = 0
    kl_sum = 0
    for idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        model.zero_grad()

        l_pw, l_qw, l_likelihood = probs(model, hyper, data, target)
        loss = ELBO(hyper, l_pw, l_qw, l_likelihood, len(loader))
        loss_sum += loss / len(loader)

        if train:
            loss.backward()
            optimizer.step()
        else:
            kl_sum += (1. / len(loader)) * (l_qw - l_pw)
    if train:
        return loss_sum
    else:
        return kl_sum

def evaluate(model, loader, infer=True, samples=1):
    acc_sum = 0
    for idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()

        if samples == 1:
            output = model(data, infer=infer)
        else:
            outputs = torch.zeros(samples, hyper.eval_batch_size, 10).cuda()
            for i in range(samples):
                outputs[i] = model(data)
            output = outputs.mean(0)

        predict = output.data.max(1)[1]
        acc = predict.eq(target.data).cpu().sum().item()
        acc_sum += acc
    return acc_sum / len(loader)


def BBB_run(hyper, train_loader, valid_loader, test_loader, n_input, n_ouput):
    """Initialize network"""
    model = BBB(n_input, n_ouput, hyper).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=hyper.lr, momentum=hyper.momentum)
    # optimizer = torch.optim.Adam(model.parameters(), lr=hyper.lr)

    train_losses = np.zeros(hyper.max_epoch)
    valid_accs = np.zeros(hyper.max_epoch)
    test_accs = np.zeros(hyper.max_epoch)

    """Train"""
    for epoch in range(hyper.max_epoch):
        train_loss = train(model, optimizer, train_loader)
        valid_acc = evaluate(model, valid_loader)
        test_acc = evaluate(model, test_loader)

        print('Epoch', epoch + 1, 'Loss', float(train_loss),
              'Valid Error', round(100 * (1 - valid_acc / hyper.eval_batch_size), 3), '%',
              'Test Error',  round(100 * (1 - test_acc / hyper.eval_batch_size), 3), '%')

        valid_accs[epoch] = valid_acc
        test_accs[epoch] = test_acc
        train_losses[epoch] = train_loss

    """Save"""
    path = 'Results/BBB_' + hyper.dataset + '_weightedll_' + str(hyper.hidden_units) + '_' + str(hyper.lr)
    wr = csv.writer(open(path + '.csv', 'w'), delimiter=',', lineterminator='\n')
    wr.writerow(['epoch', 'valid_acc', 'test_acc', 'train_losses'])

    for i in range(hyper.max_epoch):
        wr.writerow((i + 1, 1 - valid_accs[i] / hyper.eval_batch_size,
                     1 - test_accs[i] / hyper.eval_batch_size, train_losses[i]))

    torch.save(model.state_dict(), path + '.pth')

    return model


if __name__ == '__main__':
    hyper = BBB_Hyper()
    if len(sys.argv) > 1:
        hyper.hidden_units = int(sys.argv[1])

    print(hyper.hidden_units, hyper.lr, hyper.momentum, hyper.multiplier, hyper.mixture, hyper.google_init)

    """Prepare data"""
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

        n_input = 28 * 28
        n_ouput = 10
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

        n_input = 32 * 32 * 3  # 32x32 images of 3 colours
        n_ouput = 10
    else:
        raise ValueError('Unknown dataset')

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    #np.random.shuffle(indices)
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

    #model = BBB_run(hyper, train_loader, valid_loader, test_loader, n_input, n_ouput)

    '''exec(open("WeightPruning.py").read())

    """Evaluate pruned models"""
    import os
    for root, dirs, files in os.walk("Models"):
        for file in files:
            if file.startswith('BBB_MNIST2_') and file.endswith(".pth"):
                print(file)
                model1 = BBB(n_input, n_ouput, hyper)
                model1.load_state_dict(torch.load('Models/' + file))
                model1.eval()
                model1.cuda()
                print(round(1 - evaluate(model1, test_loader) / hyper.eval_batch_size, 5) * 100)'''
