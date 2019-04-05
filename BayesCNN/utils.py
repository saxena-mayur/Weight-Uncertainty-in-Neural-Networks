import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

hasGPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if hasGPU else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if hasGPU else {}


if torch.cuda.is_available():
    Var = lambda x, dtype=torch.cuda.FloatTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor
else:
    Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor
    
PI = 0.5
SIGMA_1 = torch.Tensor(np.array([np.exp(-0)])).to(DEVICE)
SIGMA_2 = torch.Tensor(np.array([np.exp(-6)])).to(DEVICE)



class Flatten(nn.Module):
    def __init__(self, nb_flat_features):
        super().__init__()
        self.nb_flat_features = nb_flat_features

    def forward(self, x):
        return x.view(-1, self.nb_flat_features)

def gaussian(x, mu, sigma):
    bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    return bell/(np.sqrt(2*np.pi)*sigma)

    
class ScaleMixtureGaussian:
    def __init__(self, PI, SIGMA_1, SIGMA_2):
        self.pi = PI
        self.sigma1 = SIGMA_1
        self.sigma2 = SIGMA_2
        
    def __call__(self, input):
        prob1 = self.pi * gaussian(input, 0., self.sigma1)
        prob2 = (1. - self.pi) * gaussian(input, 0., self.sigma2)
        return -torch.log(prob1 + prob2).sum()
    
    def __repr__(self):
        return 'ScaleMixtureGaussian(pi = {0}, sigma1 = {1}, sigma2 = {2})'.format(self.pi,
                                                               self.sigma1, self.sigma2)
    

prior_nll = ScaleMixtureGaussian(PI = 0.75,
                                 SIGMA_1 = np.exp(-1),
                                 SIGMA_2 = np.exp(-6))
