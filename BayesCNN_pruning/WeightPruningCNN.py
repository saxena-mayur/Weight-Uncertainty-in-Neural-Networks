import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

HIDDEN = 400
path = './BBB_hyperparams_400'
bayes_params = torch.load(path + '.pth')

def getThreshold(bayes_params,buckets):
    sigmas = []
    mus = []

    for name, mu, rho, sigma, eps in bayes_params:
        if 'bias' not in name:
            sigmas.append(rho.view(-1).cpu().detach().numpy())
            mus.append(mu.view(-1).cpu().detach().numpy())

    sigmas = np.concatenate(sigmas).ravel()
    mus = np.concatenate(mus).ravel()
    sigmas = np.log(1. + np.exp(sigmas))
    sign_to_noise = np.abs(mus) / sigmas
    p = np.percentile(sign_to_noise, buckets)
    
    s = np.log10(sign_to_noise)/10
    hist, bin_edges = np.histogram(s, bins='auto')
    hist = hist / s.size
    X =[]
    for i in range(hist.size):
        X.append((bin_edges[i]+bin_edges[i+1])*0.5)

    plt.plot(X,hist)
    plt.axvline(x= np.log10(p[2])/10, color='red')
    plt.ylabel('Density')
    plt.xlabel('Signal−to−Noise Ratio (dB)')
    plt.savefig('../Results/SignalToNoiseRatioDensity.png')
    plt.savefig('../Results/SignalToNoiseRatioDensity.eps', format='eps', dpi=1000)

    plt.figure(2)
    Y = np.cumsum(hist)
    plt.plot(X, Y)
    plt.axvline(x= np.log10(p[2])/10, color='red')
    plt.hlines(y= 0.75, xmin=np.min(s),xmax=np.max(s),colors='red')
    plt.ylabel('CDF')
    plt.xlabel('Signal−to−Noise Ratio (dB)')
    plt.savefig('../Results/SignalToNoiseRatioDensity_CDF.png')
    plt.savefig('../Results/SignalToNoiseRatioDensity_CDF.eps', format='eps', dpi=1000)


    return p

buckets = np.asarray([0,50,75,95,98])
thresholds = getThreshold(bayes_params,buckets)

for index in range(buckets.size):
    print(buckets[index],'-->',thresholds[index])
    t = Variable(torch.Tensor([thresholds[index]]))
    bayes_params2 = bayes_params.copy()
    for name, mu, rho, sigma, eps in bayes_params2:
        if 'bias' not in name:
            sigma = np.log(1. + np.exp(rho.cpu()))
            signalRatio = np.abs(mu.cpu()) / sigma
            signalRatio = (signalRatio > t).float() * 1
            rho.copy_((rho.cpu() * signalRatio).cuda())
            mu.copy_((mu.cpu() * signalRatio).cuda())

    torch.save(bayes_params2, path + '_Pruned_' + str(buckets[index]) + '.pth')
