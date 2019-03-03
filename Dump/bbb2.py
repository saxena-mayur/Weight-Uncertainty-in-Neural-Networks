# Drawn from https://gist.github.com/rocknrollnerd/c5af642cf217971d93f499e8f70fcb72 (in Theano)
# This is implemented in PyTorch
# Author : Anirudh Vemula

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from tqdm import tqdm, trange

from data.parser import parse_mnist
from data.dataset import MNISTDataset
from data.transforms import MNISTTransform
from torch.utils.data.dataloader import DataLoader

HIDDEN = 400
CROSS_ENTROPY=False

def log_gaussian(x, mu, sigma):
    return float(-0.5 * np.log(2 * np.pi) - np.log(np.abs(sigma))) - (x - mu)**2 / (2 * sigma**2)


def log_gaussian_logsigma(x, mu, logsigma):
    return float(-0.5 * np.log(2 * np.pi)) - logsigma - (x - mu)**2 / (2 * torch.exp(logsigma)**2)


class MLPLayer(nn.Module):
    def __init__(self, n_input, n_output, sigma_prior):
        super(MLPLayer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.sigma_prior = sigma_prior
        self.W_mu = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.01))
        self.W_logsigma = nn.Parameter(torch.Tensor(n_input, n_output).normal_(0, 0.01))
        self.b_mu = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        self.b_logsigma = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
        self.lpw = 0.
        self.lqw = 0.

    def forward(self, X, infer=False):
        if infer:
            output = torch.mm(X, self.W_mu) + self.b_mu.expand(X.size()[0], self.n_output)
            return output

        epsilon_W, epsilon_b = self.get_random()
        W = self.W_mu + torch.log(1 + torch.exp(self.W_logsigma)) * epsilon_W
        b = self.b_mu + torch.log(1 + torch.exp(self.b_logsigma)) * epsilon_b
        output = torch.mm(X, W) + b.expand(X.size()[0], self.n_output)
        self.lpw = log_gaussian(W, 0, self.sigma_prior).sum() + log_gaussian(b, 0, self.sigma_prior).sum()
        self.lqw = log_gaussian_logsigma(W, self.W_mu, self.W_logsigma).sum() + log_gaussian_logsigma(b, self.b_mu, self.b_logsigma).sum()
        return output

    def get_random(self):
        return Variable(torch.Tensor(self.n_input, self.n_output).normal_(0, self.sigma_prior).cuda()), Variable(torch.Tensor(self.n_output).normal_(0, self.sigma_prior).cuda())


class MLP(nn.Module):
    def __init__(self, n_input, sigma_prior):
        super(MLP, self).__init__()
        self.l1 = MLPLayer(n_input, HIDDEN, sigma_prior)
        self.l1_relu = nn.ReLU()
        self.l2 = MLPLayer(HIDDEN, HIDDEN, sigma_prior)
        self.l2_relu = nn.ReLU()
        self.l3 = MLPLayer(HIDDEN, 10, sigma_prior)
        self.l3_softmax = nn.Softmax(dim=1)

    def forward(self, X, infer=False):
        output = self.l1_relu(self.l1(X, infer))
        output = self.l2_relu(self.l2(output, infer))
        output = self.l3_softmax(self.l3(output, infer))
        return output

    def get_lpw_lqw(self):
        lpw = self.l1.lpw + self.l2.lpw + self.l3.lpw
        lqw = self.l1.lqw + self.l2.lqw + self.l3.lqw
        return lpw, lqw


def forward_pass_samples(X, y):
    s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.
    for _ in range(n_samples):
        output = net(X)
        sample_log_pw, sample_log_qw = net.get_lpw_lqw()
        if CROSS_ENTROPY:
            sample_log_likelihood = -100000*F.nll_loss(output, y, reduction='sum') # doesn't work
        else:
            sample_log_likelihood = log_gaussian(y, output, sigma_prior).sum() # works incredibly well
        s_log_pw += sample_log_pw
        s_log_qw += sample_log_qw
        s_log_likelihood += sample_log_likelihood

    return s_log_pw/n_samples, s_log_qw/n_samples, s_log_likelihood/n_samples


def criterion(l_pw, l_qw, l_likelihood):
    return ((1./n_batches) * (l_qw - l_pw) - l_likelihood).sum() / float(batch_size)

from six.moves import urllib
from scipy.io import loadmat
import os

mnist_path = os.path.join(".", "mnist", "mnist-original.mat")
mnist_raw = loadmat(mnist_path) # Download from https://github.com/amplab/datascience-sp14/raw/master/lab7/mldata/mnist-original.mat

N = 60000

data = np.float32(mnist_raw["data"].T[:]) / 255.
idx = np.random.choice(data.shape[0], N)
data = data[idx]
target = np.int32(mnist_raw["label"][0][idx]).reshape(N, 1)

train_idx, test_idx = train_test_split(np.array(range(N)), test_size=10000)
train_data, test_data = data[train_idx], data[test_idx]
train_target, test_target = target[train_idx], target[test_idx]

train_target = np.float32(preprocessing.OneHotEncoder(sparse=False, categories='auto').fit_transform(train_target))

n_input = train_data.shape[1] # 28*28
M = train_data.shape[0] # 50000 samples
sigma_prior = float(np.exp(-6))
n_samples = 2
learning_rate = 1e-5
n_epochs = 100

# Initialize network
net = MLP(n_input, sigma_prior)
net = net.cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

# building the objective
log_pw, log_qw, log_likelihood = 0., 0., 0.
batch_size = 125
n_batches = M / float(batch_size)
n_train_batches = int(train_data.shape[0] / float(batch_size))

test_batch_size = 1000
n_test_batches = int(test_data.shape[0] / float(test_batch_size))

train_data2, train_label, valid_data, valid_label, test_data2, test_label = parse_mnist(2, 'data/', 10000, 1)

transform = MNISTTransform()
train_dataset = MNISTDataset(train_data2, train_label, transform=transform)
valid_dataset = MNISTDataset(valid_data, valid_label, transform=transform)
test_dataset = MNISTDataset(test_data2, test_label, transform=transform)
train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, test_batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, test_batch_size, shuffle=False)

for e in tqdm(range(n_epochs)):
    errs = []

    for X, y in train_loader:
        net.zero_grad()
        if not CROSS_ENTROPY:
            y = Variable(torch.Tensor(preprocessing.OneHotEncoder(sparse=False, categories='auto').fit_transform(y.numpy().reshape(-1,1))))
        X, y = X.cuda(), y.cuda()

        log_pw, log_qw, log_likelihood = forward_pass_samples(X, y)
        loss = criterion(log_pw, log_qw, log_likelihood)
        errs.append(loss.data.cpu().numpy())
        loss.backward()
        optimizer.step()

    acc_sum = 0
    with torch.no_grad():
        for b in range(n_test_batches):
            X = Variable(torch.Tensor(test_data[b * test_batch_size: (b+1) * test_batch_size]).cuda())

            pred = net(X, infer=True)
            _, out = torch.max(pred, 1)
            eq = np.squeeze(out.data.cpu().numpy()) == np.squeeze(test_target[b * test_batch_size: (b+1) * test_batch_size])
            acc_sum += np.count_nonzero(eq)
    acc = acc_sum / (n_test_batches*test_batch_size)

    print('\nLoss', np.mean(errs), 'Error', round(100*(1 - acc), 3), '%', 'Correct', acc_sum)