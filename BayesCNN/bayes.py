import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.utils import make_grid
from torch.autograd import Variable

#Checking if gpu is available
hasGPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if hasGPU else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if hasGPU else {}

from utils import Flatten, ScaleMixtureGaussian, Var, DEVICE, PI, SIGMA_1, SIGMA_2


#Single Bayesian fully connected Layer with linear activation function  
class BayesLayer(nn.Module):
    def __init__(self, name, weight_shape, bias_shape, log_prior, functional = F.linear, **func_params):
        super().__init__()
        self.name = name
        self.weight_shape = weight_shape
        self.bias_shape = bias_shape
        self.log_prior = log_prior
        self.functional = functional
        self.kl = 0
        self.func_params = func_params

        self.weight_mu = nn.Parameter(torch.Tensor(*weight_shape).normal_(0., .1))  # or .01
        self.weight_rho = nn.Parameter(torch.Tensor(*weight_shape).uniform_(-3., -3.))  # or -4
        self.bias_mu = nn.Parameter(torch.Tensor(*bias_shape).normal_(0., .1))
        self.bias_rho = nn.Parameter(torch.Tensor(*bias_shape).uniform_(-3., -3.))
    
    def sample(self):
        weight_sigma = torch.log(1. + torch.exp(self.weight_rho))
        bias_sigma = torch.log(1. + torch.exp(self.bias_rho))
        
        # Sample weights and bias
        epsilon_weight = torch.randn_like(self.weight_mu).to(DEVICE)
        epsilon_bias = torch.randn_like(self.bias_mu).to(DEVICE)
        weight = self.weight_mu + weight_sigma * epsilon_weight
        bias = self.bias_mu + bias_sigma * epsilon_bias
        
        self.kl = (-.5*torch.log(np.pi*np.e*weight_sigma.pow(2)).sum()
                   -.5*torch.log(np.pi*np.e*bias_sigma.pow(2)).sum() 
                   - self.log_prior(weight) 
                   - self.log_prior(bias))
        return weight, bias
    
    def forward(self, input, noise = True):
        if not noise:
            return self.functional(input, self.weight_mu, self.bias_mu)
        weight, bias = self.sample()
        return self.functional(input, weight, bias, **self.func_params)
    
    def __repr__(self):
        return '{0}(weight_shape={1}, bias_shape={2})'.format(
                        self.name, self.weight_shape, self.bias_shape)
        
        
    
class Bayesian(nn.Sequential):
    def __init__(self, net, log_prior):
        self.log_prior = log_prior
        super().__init__(*[self._bayesianize(l) for l in net])
        
    def _bayesianize(self, layer):
        if isinstance(layer, nn.Linear):
            return BayesLayer(
                        name = 'BayesLinear',
                        weight_shape = (layer.out_features, layer.in_features),
                        bias_shape = (layer.out_features, ), 
                        log_prior = self.log_prior, 
                        functional = F.linear)
        
        elif isinstance(layer, nn.Conv2d):
            return BayesLayer(
                        name = 'BayesConv2D',
                        weight_shape = (layer.out_channels,
                                        layer.in_channels, 
                                        *layer.kernel_size), 
                        bias_shape = (layer.out_channels,), 
                        log_prior = self.log_prior, 
                        functional = F.conv2d, 
                        padding = layer.padding, 
                        stride = layer.stride, 
                        dilation = layer.dilation, 
                        groups = layer.groups)

        return layer
 
    def kl(self):
        return sum([l.kl for l in self if hasattr(l, 'kl')])
        


