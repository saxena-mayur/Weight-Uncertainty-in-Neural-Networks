import csv
import sys
import math

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
        self.dataset = 'mnist'  # mnist || cifar10 || fmnist

        self.lr = 1e-4
        self.momentum = 0.95
        self.hidden_units = 400
        self.mixture = True
        self.pi = 0.75
        self.s1 = float(np.exp(-8))
        self.s2 = float(np.exp(-1))
        self.rho_init = -8
        self.multiplier = 1.

        self.max_epoch = 600
        self.n_samples = 1
        self.n_test_samples = 10
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

        # We initialise weigth_mu and bias_mu as for usual Linear layers in PyTorch
        self.weight_mu = nn.Parameter(torch.Tensor(n_output, n_input))
        self.bias_mu = nn.Parameter(torch.Tensor(n_output))

        torch.nn.init.kaiming_uniform_(self.weight_mu, nonlinearity='relu')
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias_mu, -bound, bound)

        self.bias_rho = nn.Parameter(torch.Tensor(n_output).normal_(hyper.rho_init, .05))
        self.weight_rho = nn.Parameter(torch.Tensor(n_output, n_input).normal_(hyper.rho_init, .05))

        self.lpw = 0.
        self.lqw = 0.

    def forward(self, data, infer=False):
        if infer:
            output = F.linear(data, self.weight_mu, self.bias_mu)
            return output

        epsilon_W = Variable(torch.Tensor(self.n_output, self.n_input).normal_(0, 1).cuda())
        epsilon_b = Variable(torch.Tensor(self.n_output).normal_(0, 1).cuda())
        W = self.weight_mu + torch.log(1+torch.exp(self.weight_rho)) * epsilon_W
        b = self.bias_mu + torch.log(1+torch.exp(self.bias_rho)) * epsilon_b

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
        output = F.softmax(self.layers[2](output, infer), dim=1)
        return output

    def get_lpw_lqw(self):
        lpw = self.layers[0].lpw + self.layers[1].lpw + self.layers[2].lpw
        lqw = self.layers[0].lqw + self.layers[1].lqw + self.layers[2].lqw
        return lpw, lqw


def probs(model, hyper, data, target):
    s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
    for _ in range(hyper.n_samples):
        output = torch.log(model(data))

        sample_log_pw, sample_log_qw = model.get_lpw_lqw()
        sample_log_likelihood = -F.nll_loss(output, target, reduction='sum') * hyper.multiplier

        s_log_pw += sample_log_pw / hyper.n_samples
        s_log_qw += sample_log_qw / hyper.n_samples
        s_log_likelihood += sample_log_likelihood / hyper.n_samples

    return s_log_pw, s_log_qw, s_log_likelihood

def ELBO(l_pw, l_qw, l_likelihood, beta):
    kl = beta * (l_qw - l_pw)
    return kl - l_likelihood


def train(model, optimizer, loader, train=True):
    loss_sum = 0
    kl_sum = 0
    m = len(loader)

    for batch_id, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        model.zero_grad()

        #beta = 2 ** (m - (batch_id + 1)) / (2 ** m - 1)
        beta = 1 / (m)

        l_pw, l_qw, l_likelihood = probs(model, hyper, data, target)
        loss = ELBO(l_pw, l_qw, l_likelihood, beta)
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
            output = model(data)
            for i in range(samples - 1):
                output += model(data)

        predict = output.data.max(1)[1]
        acc = predict.eq(target.data).cpu().sum().item()
        acc_sum += acc
    return acc_sum / len(loader)


def BBB_run(hyper, train_loader, valid_loader, test_loader, n_input, n_ouput, id=0):
    print(hyper.__dict__)

    """Initialize network"""
    model = BBB(n_input, n_ouput, hyper).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=hyper.lr, momentum=hyper.momentum)

    train_losses = np.zeros(hyper.max_epoch)
    valid_accs = np.zeros(hyper.max_epoch)
    test_accs = np.zeros(hyper.max_epoch)

    """Train"""
    for epoch in range(hyper.max_epoch):
        train_loss = train(model, optimizer, train_loader)
        valid_acc = evaluate(model, valid_loader, samples=hyper.n_test_samples)
        test_acc = evaluate(model, test_loader, samples=hyper.n_test_samples)

        print('Epoch', epoch + 1, 'Loss', float(train_loss),
              'Valid Error', round(100 * (1 - valid_acc / hyper.eval_batch_size), 3), '%',
              'Test Error',  round(100 * (1 - test_acc / hyper.eval_batch_size), 3), '%')

        valid_accs[epoch] = valid_acc
        test_accs[epoch] = test_acc
        train_losses[epoch] = train_loss

    """Save"""
    path = 'Results/BBB_' + hyper.dataset + '_' + str(hyper.hidden_units) + '_' + str(hyper.lr) + '_samples' + str(hyper.n_samples)+ '_ID' + str(id)
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

    if len(sys.argv) > 2:
        hyper.dataset = sys.argv[2]

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
    elif hyper.dataset == 'fmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255. / 126.),  # divide as in paper
        ])

        train_data = datasets.FashionMNIST(
            root='data2',
            train=True,
            download=True,
            transform=transform)
        test_data = datasets.FashionMNIST(
            root='data2',
            train=False,
            download=True,
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

    # Results of hyperparameter selection for mixture priors
    top = {
        400: [
        		(0.25, -8, 1, 6)
             ],
        800: [
        		(0.25, -7, 0, 8)
              ],
        1200: [
        		(0.25, -7, 1, 8)
              ]
    }
    # Test params
    pi, rho, s1, s2 = top[hyper.hidden_units][0]
    hyper.pi = pi
    hyper.rho_init = rho
    hyper.s1 = float(np.exp(-s1))
    hyper.s2 = float(np.exp(-s2))
    model = BBB_run(hyper, train_loader, valid_loader, test_loader, n_input, n_ouput)

    #exec(open("WeightPruning.py").read())

    """Evaluate pruned models"""
    '''import os
    for root, dirs, files in os.walk("Results/400/"):
        for file in files:
            if file.startswith('BBB2_mnist_400_0.0001_samples1_ID4_Pruned_98') and file.endswith(".pth"):
                print(file)
                model1 = BBB(n_input, n_ouput, hyper)
                model1.load_state_dict(torch.load('Results/400/' + file))
                model1.eval()
                model1.cuda()
                print('Valid', round(1 - evaluate(model1, valid_loader) / hyper.eval_batch_size, 5) * 100)
                print('Valid 10', round(1 - evaluate(model1, valid_loader, samples=10) / hyper.eval_batch_size, 5) * 100)
                #print('Test', round(1 - evaluate(model1, test_loader) / hyper.eval_batch_size, 5) * 100)
                #print('Test 10', round(1 - evaluate(model1, test_loader, samples=10) / hyper.eval_batch_size, 5) * 100)'''