import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

#Checking if gpu is available
hasGPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if hasGPU else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if hasGPU else {}

from utils import Flatten, ScaleMixtureGaussian, Var, DEVICE, PI, SIGMA_1, SIGMA_2, gaussian

class BayesWrapper:
    def __init__(self, name, net, prior_nll, 
                 rho_init=-5,
                 mode='classification', lr = 1e-3):
        super().__init__()
        self.name = name # network name
        self.net = net
        self.bayes_params = [(name,
                              p.clone().detach(), #mu
                              torch.zeros_like(p)+rho_init,#rho,
                              torch.zeros_like(p),#sigma
                              torch.zeros_like(p)) #epsilon (buffer)
                             for name,p in self.net.named_parameters()
                            ]
        for (name, *tensors) in self.bayes_params:
            for t in tensors:
                t.to(DEVICE)
        self.kl, self.pnll, self.vp = 0, 0, 0
        self.prior_nll = prior_nll
        if mode == 'regression':
            self.criterion = lambda x, y:.5*((x-y)**2).sum()
        elif mode == 'classification':
            self.criterion = nn.CrossEntropyLoss()
        params = [mu for name, mu, rho, _, eps in self.bayes_params]+ [rho for name, mu, rho, _, eps in self.bayes_params]
        #self.optimizer = optim.SGD(params, lr=lr)
        self.optimizer = optim.Adam(params, lr=lr)
    
    def forward(self, input):     
        for name, mu, rho, sigma, eps in self.bayes_params:
            eps.normal_()
            sigma.copy_(torch.log1p(torch.exp(rho)))
            w = mu + eps*sigma
            self.pnll += self.prior_nll(w)
            self.vp += (-torch.log(np.sqrt(2*np.pi)*sigma) - eps**2/2).sum()
            self.net.load_state_dict(OrderedDict({name:w}), strict=False)
        return self.net(input)


    def step(self, outputs, targets, beta):
        self.net.zero_grad()
        n_samples = len(outputs)
        xe = sum([self.criterion(out, targets) for out in outputs])/n_samples
        kl = (self.vp + self.pnll)/n_samples
        net_loss = xe + beta*self.pnll/n_samples
        loss = net_loss + beta*self.vp/n_samples
        net_loss.backward() # with respect to w
        
        for (name, mu, rho, sigma, eps), p in zip(self.bayes_params, self.net.parameters()):
            mu.grad = p.grad
            rho.grad = eps*(p.grad - beta/sigma)/(1+torch.exp(-rho))
            
        self.optimizer.step()
        self.vp, self.pnll, self.kl = 0, 0, 0
        return kl, xe, loss
        
    def __repr__(self):
        return 'BayesWrapper(\n{0})'.format(self.net.__repr__())

    def __call__(self, input):
        return self.forward(input)
    
    def train(self):
        self.net.train()