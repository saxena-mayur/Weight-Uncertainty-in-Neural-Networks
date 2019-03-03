import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from tqdm import tqdm

from data.parser import parse_mnist
from data.dataset import MNISTDataset
from data.transforms import MNISTTransform
from torch.utils.data.dataloader import DataLoader

hasGPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if hasGPU else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if hasGPU else {}

class Config(object):

    def __init__(self):
        # Network Layout
        self.n_input = 28 * 28
        self.classes = 10
        self.hidden_units = 400

        # Dataset
        self.batch_size = 125
        self.eval_batch_size = 1000

        # Hyperparameters
        self.learning_rate = 1e-3
        self.pi = 0.25
        self.s1 = 0.75
        self.s2 = 0.1
        self.train_samples = 1
        self.test_samples = 1
        self.n_epoch = 100


GAUSSIAN_SCALER = 1. / np.sqrt(2.0 * np.pi)
def gaussian(x, mu, sigma):
    bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    return torch.clamp(GAUSSIAN_SCALER / sigma * bell, 1e-10, 1.)  # clip

def prior(input, pi, s1, s2):
    prob1 = pi * gaussian(input, 0., s1)
    prob2 = (1. - pi) * gaussian(input, 0., s2)
    return torch.log(prob1 + prob2)

class BBBLayer(nn.Module):
    def __init__(self, n_input, n_output, cfg):
        super(BBBLayer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.cfg = cfg

        # Initialise weights and bias
        self.W_mu = nn.Parameter(torch.Tensor(n_output, n_input).normal_(0., .1))
        self.W_rho = nn.Parameter(torch.Tensor(n_output, n_input).uniform_(-3., -3.))
        self.b_mu = nn.Parameter(torch.Tensor(n_output).normal_(0., .1))
        self.b_rho = nn.Parameter(torch.Tensor(n_output).uniform_(-3., -3.))

        # Initiliase prior and posterior
        self.lpw = 0.
        self.lqw = 0.

    def forward(self, input, infer=False):
        if infer:
            return F.linear(input, self.W_mu, self.b_mu)

        # Obtain positive sigma from logsigma, as in paper
        W_sigma = torch.log(1. + torch.exp(self.W_rho))
        b_sigma = torch.log(1. + torch.exp(self.b_rho))

        # Sample weights and bias
        epsilon_W = Variable(torch.Tensor(self.n_output, self.n_input).normal_(0., 1.)).to(DEVICE)
        epsilon_b = Variable(torch.Tensor(self.n_output).normal_(0., 1.)).to(DEVICE)
        W = self.W_mu + W_sigma * epsilon_W
        b = self.b_mu + b_sigma * epsilon_b

        # Compute posterior and prior probabilities
        self.lpw = prior(W, cfg.pi, cfg.s1, cfg.s2).sum() + prior(b, cfg.pi, cfg.s1, cfg.s2).sum()
        self.lqw = torch.log(gaussian(W, self.W_mu, W_sigma)).sum() + torch.log(gaussian(b, self.b_mu, b_sigma)).sum()

        # Pass sampled weights and bias on to linear layer
        return F.linear(input, W, b)


class BBB(nn.Module):
    def __init__(self, cfg):
        super(BBB, self).__init__()
        self.cfg = cfg
        self.l1 = BBBLayer(cfg.n_input, cfg.hidden_units, cfg)
        self.l1_relu = nn.ReLU()
        self.l2 = BBBLayer(cfg.hidden_units, cfg.hidden_units, cfg)
        self.l2_relu = nn.ReLU()
        self.l3 = BBBLayer(cfg.hidden_units, cfg.classes, cfg)
        self.l3_logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, X, infer=False):
        output = self.l1_relu(self.l1.forward(X, infer))
        output = self.l2_relu(self.l2.forward(output, infer))
        output = self.l3_logsoftmax(self.l3.forward(output, infer))
        return output

    def get_lpw_lqw(self):
        lpw = self.l1.lpw + self.l2.lpw + self.l3.lpw
        lqw = self.l1.lqw + self.l2.lqw + self.l3.lqw
        return lpw, lqw


def BBB_loss(input, target, cfg):
    s_log_pw, s_log_qw, s_log_likelihood = 0., 0., 0.

    for _ in range(cfg.train_samples):
        output = model.forward(input)
        sample_log_pw, sample_log_qw = model.get_lpw_lqw()
        sample_log_likelihood = -F.nll_loss(output, target, reduction='sum')
        s_log_pw += sample_log_pw
        s_log_qw += sample_log_qw
        s_log_likelihood += sample_log_likelihood

    l_pw, l_qw, l_likelihood =  s_log_pw / cfg.train_samples, s_log_qw / cfg.train_samples, s_log_likelihood / cfg.train_samples

    return (1. / len(train_loader)) * (l_qw - l_pw) - l_likelihood

torch.manual_seed(0) # for reproducibility
cfg = Config()

# Load MNIST data into dataloaders, with correct batch size
train_data, train_label, valid_data, valid_label, test_data, test_label = parse_mnist()
train_dataset = MNISTDataset(train_data, train_label, transform=MNISTTransform())
valid_dataset = MNISTDataset(valid_data, valid_label, transform=MNISTTransform())
test_dataset = MNISTDataset(test_data, test_label, transform=MNISTTransform())
train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, **LOADER_KWARGS)
valid_loader = DataLoader(valid_dataset, cfg.eval_batch_size, shuffle=False, **LOADER_KWARGS)
test_loader = DataLoader(test_dataset, cfg.eval_batch_size, shuffle=False, **LOADER_KWARGS)

# Initialize network
model = BBB(cfg).to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)

for e in tqdm(range(cfg.n_epoch)):
    errs = []

    for batch_idx, (input, target) in enumerate(train_loader):
      model.zero_grad()
      input, target = input.to(DEVICE), target.to(DEVICE)
      loss = BBB_loss(input, target, cfg)
      errs.append(loss.data.cpu().numpy())
      loss.backward()
      optimizer.step()

    acc_sum = 0.
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(valid_loader):
            input, target = input.to(DEVICE), target.to(DEVICE)
            output = model.forward(input, infer=True)
            predict = output.data.max(1)[1]
            acc_sum += predict.eq(target.data).cpu().sum().item()
    acc = acc_sum / len(valid_data)

    print('Epoch', e + 1, 'Loss', np.mean(errs), 'Error', round(100*(1 - acc), 3), '%', 'Correct', acc_sum)