import csv

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
        self.hidden_units = 400
        self.s1 = float(np.exp(-6))
        self.s2 = float(np.exp(-1))
        self.pi = 0.75

        self.mixture = False
        self.google_init = False

        self.max_epoch = 40
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
            self.W_mu = nn.Parameter(torch.Tensor(n_output, n_input).normal_(0., .05))  # or .01
            self.W_rho = nn.Parameter(torch.Tensor(n_output, n_input).normal_(-5., .05))  # or -4
            self.b_mu = nn.Parameter(torch.Tensor(n_output).normal_(0., .05))
            self.b_rho = nn.Parameter(torch.Tensor(n_output).normal_(-5., .05))

        else:
            self.W_mu = nn.Parameter(torch.Tensor(n_output, n_input).normal_(0, 0.01))
            self.W_rho = nn.Parameter(torch.Tensor(n_output, n_input).normal_(0, 0.01))
            self.b_mu = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))
            self.b_rho = nn.Parameter(torch.Tensor(n_output).uniform_(-0.01, 0.01))

        self.lpw = 0.
        self.lqw = 0.

    def forward(self, data, infer=False):
        if infer:
            output = F.linear(data, self.W_mu, self.b_mu)
            return output

        epsilon_W = Variable(torch.Tensor(self.n_output, self.n_input).normal_(0, self.s1).cuda())
        epsilon_b = Variable(torch.Tensor(self.n_output).normal_(0, self.s1).cuda())
        W = self.W_mu + torch.log1p(torch.exp(self.W_rho)) * epsilon_W
        b = self.b_mu + torch.log1p(torch.exp(self.b_rho)) * epsilon_b

        output = F.linear(data, W, b)

        self.lqw = log_gaussian_rho(W, self.W_mu, self.W_rho).sum() + \
                   log_gaussian_rho(b, self.b_mu, self.b_rho).sum()
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

        self.l1 = BBBLayer(n_input, hyper.hidden_units, hyper)
        self.l2 = BBBLayer(hyper.hidden_units, hyper.hidden_units, hyper)
        self.l3 = BBBLayer(hyper.hidden_units, n_ouput, hyper)

    def forward(self, data, infer=False):
        output = F.relu(self.l1(data.view(-1, self.n_input), infer))
        output = F.relu(self.l2(output, infer))
        output = F.log_softmax(self.l3(output, infer), dim=1)
        return output

    def get_lpw_lqw(self):
        lpw = self.l1.lpw + self.l2.lpw + self.l3.lpw
        lqw = self.l1.lqw + self.l2.lqw + self.l3.lqw
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


def evaluate(model, loader, infer=True):
    acc_sum = 0
    for idx, (data, target) in enumerate(loader):
        data, target = data.cuda(), target.cuda()
        output = model(data, infer=infer)
        predict = output.data.max(1)[1]
        acc = predict.eq(target.data).cpu().sum().item()
        acc_sum += acc
    return acc_sum / len(loader)


def BBB_run(hyper):
    """Prepare data"""
    valid_size = 1 / 6

    if hyper.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 255. / 126.), # divide as in paper
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
    np.random.shuffle(indices)
    split = int(valid_size * num_train)
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=hyper.batch_size,
        sampler=train_sampler,
        num_workers=2)
    valid_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=hyper.eval_batch_size,
        sampler=valid_sampler,
        num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=hyper.eval_batch_size,
        num_workers=2)

    """Initialize network"""
    model = BBB(n_input, n_ouput, hyper).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=hyper.lr, momentum=hyper.momentum)

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
    path = 'Results/SGD_' + hyper.dataset + '_' + str(hyper.hidden_units) + '_' + str(hyper.lr)
    wr = csv.writer(open(path + '.csv', 'w'), delimiter=',', lineterminator='\n')
    wr.writerow(['epoch', 'valid_acc', 'test_acc', 'train_losses'])

    for i in range(hyper.max_epoch):
        wr.writerow((i + 1, 1 - valid_accs[i] / hyper.eval_batch_size,
                     1 - test_accs[i] / hyper.eval_batch_size, train_losses[i]))

    torch.save(model.state_dict(), path + '.pth')

    return model


if __name__ == '__main__':
    hyper = BBB_Hyper()
    print(hyper.hidden_units, hyper.n_samples, hyper.lr, hyper.momentum, hyper.multiplier, hyper.mixture)
    model = BBB_run(hyper)
