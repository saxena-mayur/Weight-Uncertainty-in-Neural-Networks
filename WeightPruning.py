from BayesBackpropagation import *
import numpy as np
import copy
import matplotlib.pyplot as plt
plt.switch_backend('agg')

hasGPU = False
DEVICE = torch.device("cuda" if hasGPU else "cpu")

# Hyperparameter declaration
BATCH_SIZE = 125
TEST_BATCH_SIZE = 1000
CLASSES = 10
TRAIN_EPOCHS = 100
SAMPLES = 1
PI = 0.25
SIGMA_1 = torch.FloatTensor([0.75]).to(DEVICE)
SIGMA_2 = torch.FloatTensor([0.1]).to(DEVICE)
INPUT_SIZE = 28 * 28
LAYERS = np.array([400, 400])
NUM_BATCHES = 0
ACTIVATION_FUNCTIONS = np.array(['relu','relu','softmax'])
model = BayesianNetwork(inputSize=INPUT_SIZE,
                        CLASSES=CLASSES,
                        layers=LAYERS,
                        activations=ACTIVATION_FUNCTIONS,
                        SAMPLES=SAMPLES,
                        BATCH_SIZE=BATCH_SIZE,
                        NUM_BATCHES=NUM_BATCHES,
                        hasScalarMixturePrior=True,
                        PI=PI,
                        SIGMA_1=SIGMA_1,
                        SIGMA_2=SIGMA_2).to(DEVICE)

model.load_state_dict(torch.load('./Models/BBB_MNIST_400_5samples.pth'))
model.eval()

def getThreshold(model,buckets):
    sigmas = []
    mus = []

    for i in range(3):
        sigmas.append(model.state_dict()['layers.'+str(i)+'.weight_rho'].view(-1).cpu().detach().numpy())
        mus.append(model.state_dict()['layers.'+str(i)+'.weight_mu'].view(-1).cpu().detach().numpy())

    sigmas = np.concatenate(sigmas).ravel()
    mus = np.concatenate(mus).ravel()
    sigmas = np.log(1. + np.exp(sigmas))
    sign_to_noise = np.abs(mus) / sigmas
    p = np.percentile(sign_to_noise, buckets)
    
    s = np.log10(sign_to_noise)/10
    hist, bin_edges = np.histogram(s, bins='auto', density=True)
    X =[]
    for i in range(10):
        X.append((bin_edges[i]+bin_edges[i+1])*0.5)
    
    plt.plot(X,hist)
    plt.axvline(x= np.log10(p[2])/10, color='red')
    plt.ylabel('Density')
    plt.xlabel('Signal−to−Noise Ratio (dB)')
    plt.savefig('./Results/SignalToNoiseRatioDensity.png')

    plt.figure(2)
    Y = np.cumsum(hist)
    plt.plot(X, Y)
    plt.axvline(x= np.log10(p[2])/10, color='red')
    plt.hlines(y= 0.75, xmin=np.min(s),xmax=np.max(s),colors='red')
    plt.ylabel('CDF')
    plt.xlabel('Signal−to−Noise Ratio (dB)')
    plt.savefig('./Results/SignalToNoiseRatioDensity_CDF.png')
    
    return p

buckets = np.asarray([0,50,75,95,98])
thresholds = getThreshold(model,buckets)

for index in range(buckets.size):
    print(buckets[index],'-->',thresholds[index])
    t = Variable(torch.Tensor([thresholds[index]]))
    model1 = copy.deepcopy(model)
    for i in range(3):
        sigma = model.state_dict()['layers.'+str(i)+'.weight_rho']
        mu = model.state_dict()['layers.'+str(i)+'.weight_mu'] 
        sigma = np.log(1. + np.exp(sigma))
        signalRatio = np.abs(mu) / sigma
        signalRatio = (signalRatio > t).float() * 1
        model1.state_dict()['layers.'+str(i)+'.weight_rho'].data.copy_(sigma * signalRatio)
        model1.state_dict()['layers.'+str(i)+'.weight_mu'].data.copy_(mu * signalRatio)

    torch.save(model1.state_dict(), './Models/BBB_MNIST_Pruned_'+str(buckets[index])+'.pth')
