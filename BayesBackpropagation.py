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

#Gaussian class
class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho)) #as mentioned in paper, to make sigma always positive
    
    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(DEVICE)
        return self.mu + self.sigma * epsilon
    
    def log_prob(self, input):
        return (-math.log(math.sqrt(2 * math.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()

#Mixing two gaussians based on a probability
class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)
    
    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum() #scalar mixing

#Single Bayesian fully connected Layer with linear activation function  
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, parent):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))#.normal_(0, 0.01))#
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))# .normal_(0, 0.01))#
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))#.uniform_(-0.01, 0.01))#
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))#.uniform_(-0.01, 0.01))#
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        if parent.hasScalarMixturePrior == True:
            self.weight_prior = ScaleMixtureGaussian(parent.PI, parent.SIGMA_1, parent.SIGMA_2)
            self.bias_prior = ScaleMixtureGaussian(parent.PI, parent.SIGMA_1, parent.SIGMA_2)
        else:
            self.weight_prior = Gaussian(0, parent.SIGMA_1)
            self.bias_prior = Gaussian(0, parent.SIGMA_1)
        self.log_prior = 0.
        self.log_variational_posterior = 0.
        self.SIGMA_1 = parent.SIGMA_1
        self.SIGMA_2 = parent.SIGMA_2

    #Forward propagation
    def forward(self, input, sample=False, calculate_log_probs=False):
        
        #epsilon_W, epsilon_b = self.get_random()
        if self.training or sample:
            weight = self.weight.sample() #* epsilon_W
            bias = self.bias.sample() #* epsilon_b
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0

        return F.linear(input, weight, bias)
    
    def get_random(self):
        return Variable(torch.Tensor(self.in_features).normal_(0, float(self.SIGMA_1)).to(DEVICE)),\
                 Variable(torch.Tensor(self.out_features).normal_(0, float(self.SIGMA_1)).to(DEVICE))

class BayesianNetwork(nn.Module):
    def __init__(self, inputSize, CLASSES, layers, activations, SAMPLES, BATCH_SIZE, NUM_BATCHES, hasScalarMixturePrior, PI, SIGMA_1, SIGMA_2):
        super().__init__()
        self.inputSize = inputSize
        self.activations = activations
        self.CLASSES = CLASSES
        self.SAMPLES = SAMPLES
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_BATCHES = NUM_BATCHES
        self.DEPTH = 0 #captures depth of network
        #to make sure that number of hidden layers is one less than number of activation function
        assert (activations.size - layers.size) == 1

        self.SIGMA_1 = SIGMA_1
        self.hasScalarMixturePrior = hasScalarMixturePrior
        if hasScalarMixturePrior == True:
            self.SIGMA_2 = SIGMA_2
            self.PI = PI

        self.layers = nn.ModuleList([]) #To combine consecutive layers
        if layers.size == 0:
            self.layers.append(BayesianLinear(inputSize, CLASSES, self))
            self.DEPTH += 1
        else:
            self.layers.append(BayesianLinear(inputSize, layers[0], self))
            self.DEPTH += 1
            for i in range(layers.size-1):
                self.layers.append(BayesianLinear(layers[i], layers[i+1], self))
                self.DEPTH += 1
            self.layers.append(BayesianLinear(layers[layers.size-1], CLASSES, self)) #output layer
            self.DEPTH += 1
    
    #Forward propagation and assigning activation functions to linear layers
    def forward(self, x, sample=False):
        x = x.view(-1, self.inputSize)
        layerNumber = 0
        for i in range(self.activations.size):
            if self.activations[i]=='relu':
                x = F.relu(self.layers[layerNumber](x, sample))
            elif self.activations[i]=='softmax':
                x = F.log_softmax(self.layers[layerNumber](x, sample), dim=1)
            else:
                x = self.layers[layerNumber](x, sample)
            layerNumber+= 1
        return x
    
    def log_prior(self): # p(w) in paper
        value = 0.
        for i in range(self.DEPTH):
            value+= self.layers[i].log_prior
        return value
    
    def log_variational_posterior(self): # q(w) in paper
        value = 0.
        for i in range(self.DEPTH):
            value+= self.layers[i].log_variational_posterior
        return value
    """
    def sample_elbo(self, input, target):
        # Reserve memory in GPU for computations
        outputs = torch.zeros(self.SAMPLES, self.BATCH_SIZE, self.CLASSES).to(DEVICE)
        qws = torch.zeros(self.SAMPLES).to(DEVICE)
        pws = torch.zeros(self.SAMPLES).to(DEVICE)
        ll = torch.zeros(self.SAMPLES).to(DEVICE)
        
        for i in range(self.SAMPLES):
            outputs[i] = self.forward(input, sample=True)
            pws[i] = self.log_prior() # p(w) in paper
            qws[i] = self.log_variational_posterior() # q(w) in paper
            if self.CLASSES == 1: # quadratic loss for regression TODO: WHY NOT LOG?
                ll[i] = -(.5 * (target - outputs[i]) ** 2).sum()
        
        # Aggregate results
        pw = pws.mean()
        qw = pws.mean()
        
        if self.CLASSES == 1:
            ll = ll.mean()
        else:
            ll = - F.nll_loss(outputs.mean(0), target, reduction='sum')
        
        loss = (qw - pw)/self.NUM_BATCHES - ll
        
        return loss, pw, qw, ll
    """
    def sample_elbo(self, input, target):
        samples=self.SAMPLES
        outputs = torch.zeros(samples, self.BATCH_SIZE, self.CLASSES).to(DEVICE)
        log_priors = torch.zeros(samples).to(DEVICE)
        log_variational_posteriors = torch.zeros(samples).to(DEVICE)
        negative_log_likelihood = torch.zeros(samples).to(DEVICE)
        
        for i in range(samples):
            outputs[i] = self.forward(input, sample=True)
            log_priors[i] = self.log_prior()
            log_variational_posteriors[i] = self.log_variational_posterior()
            if self.CLASSES == 1:
                negative_log_likelihood[i] = (.5 * (target - outputs[i]) ** 2).sum()
            
        log_prior = log_priors.mean()
        log_variational_posterior = log_variational_posteriors.mean()
        if self.CLASSES > 1:
            negative_log_likelihood = negative_log_likelihood.sum()#F.nll_loss(outputs.mean(0), target, reduction='sum')
        else:
            negative_log_likelihood = negative_log_likelihood.mean()
        loss = (log_variational_posterior - log_prior)/self.NUM_BATCHES + negative_log_likelihood
        return loss, log_prior, log_variational_posterior, negative_log_likelihood