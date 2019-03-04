from tqdm import tqdm
import math

from BayesBackpropagation import *

from torch.utils.data.dataloader import DataLoader

from data.parser import parse_mnist
from data.dataset import MNISTDataset
from data.transforms import MNISTTransform

import torch.optim as optim


class MNIST(object):
    def __init__(self, BATCH_SIZE, TEST_BATCH_SIZE, CLASSES, TRAIN_EPOCHS, SAMPLES, hasScalarMixturePrior, PI, SIGMA_1,
                 SIGMA_2, INPUT_SIZE, LAYERS, ACTIVATION_FUNCTIONS, LR, MODE='mlp'):
        # Prepare data
        if MODE == 'mlp':
            train_data, train_label, valid_data, valid_label, test_data, test_label = parse_mnist(2)
        elif MODE == 'cnn':
            train_data, train_label, valid_data, valid_label, test_data, test_label = parse_mnist(4)
        else:
            raise ValueError('Usupported mode')

        train_dataset = MNISTDataset(train_data, train_label, transform=MNISTTransform())
        valid_dataset = MNISTDataset(valid_data, valid_label, transform=MNISTTransform())
        test_dataset = MNISTDataset(test_data, test_label, transform=MNISTTransform())
        self.train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, **LOADER_KWARGS)
        self.valid_loader = DataLoader(valid_dataset, TEST_BATCH_SIZE, shuffle=False, **LOADER_KWARGS)
        self.test_loader = DataLoader(test_dataset, TEST_BATCH_SIZE, shuffle=False, **LOADER_KWARGS)

        # Hyper parameter setting
        self.TEST_BATCH_SIZE = TEST_BATCH_SIZE
        self.TRAIN_SIZE = len(self.train_loader.dataset)
        self.TEST_SIZE = len(self.test_loader.dataset)
        self.NUM_BATCHES = len(self.train_loader)
        self.NUM_TEST_BATCHES = len(self.test_loader)

        self.CLASSES = CLASSES
        self.TRAIN_EPOCHS = TRAIN_EPOCHS
        self.SAMPLES = SAMPLES

        # Checking if the mentioned batch sizes are feasible
        assert (self.TRAIN_SIZE % BATCH_SIZE) == 0
        assert (self.TEST_SIZE % TEST_BATCH_SIZE) == 0

        # Network Declaration
        self.net = BayesianNetwork(inputSize=INPUT_SIZE, \
                                   CLASSES=CLASSES, \
                                   layers=LAYERS, \
                                   activations=ACTIVATION_FUNCTIONS, \
                                   SAMPLES=SAMPLES, \
                                   BATCH_SIZE=BATCH_SIZE, \
                                   NUM_BATCHES=self.NUM_BATCHES, \
                                   hasScalarMixturePrior=hasScalarMixturePrior, \
                                   PI=PI, \
                                   SIGMA_1=SIGMA_1, \
                                   SIGMA_2=SIGMA_2).to(DEVICE)

        # Optimizer declaration
        self.optimizer = optim.SGD(self.net.parameters(), lr=LR)  # self.optimizer = optim.Adam(self.net.parameters())

    # Define the training step for MNIST data set
    def train(self):
        loss = 0.
        for batch_idx, (input, target) in enumerate(self.train_loader):
            input, target = input.to(DEVICE), target.to(DEVICE)
            self.net.zero_grad()
            loss = self.net.BBB_loss(input, target)
            loss.backward()
            self.optimizer.step()
        return loss

    # Testing the ensemble
    def test(self, valid=True):
        data_loader = self.valid_loader if valid else self.test_loader
        correct = 0
        with torch.no_grad():
            for input, target in data_loader:
                input, target = input.to(DEVICE), target.to(DEVICE)
                output = self.net.forward(input, infer=True)
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / self.TEST_SIZE

        return round(100 * (1 - accuracy), 3)  # Error

# Multiple epochs
def multipleEpochAnalyis():
    #Hyperparameter declaration
    BATCH_SIZE = 125
    TEST_BATCH_SIZE = 1000
    CLASSES = 10
    TRAIN_EPOCHS = 100
    SAMPLES = 2
    PI = 0.25
    SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
    SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])
    #SIGMA_1 = torch.cuda.FloatTensor([0.75])
    #SIGMA_2 = torch.cuda.FloatTensor([0.1])
    INPUT_SIZE = 28*28
    LAYERS = np.array([400,400])

    errorRate = [] #to store error rates at different epochs

    mnist = MNIST(BATCH_SIZE = BATCH_SIZE,\
                TEST_BATCH_SIZE = TEST_BATCH_SIZE,\
                CLASSES = CLASSES,\
                TRAIN_EPOCHS = TRAIN_EPOCHS,\
                SAMPLES = SAMPLES,\
                hasScalarMixturePrior = True,\
                PI = PI,\
                SIGMA_1 = SIGMA_1,\
                SIGMA_2 = SIGMA_2,\
                INPUT_SIZE = INPUT_SIZE,\
                LAYERS = LAYERS,\
                ACTIVATION_FUNCTIONS = np.array(['relu','relu','softmax']),
                LR=1e-3)

    for _ in tqdm(range(TRAIN_EPOCHS)):
        loss = mnist.train()
        err = mnist.test()
        print(err, float(loss))
        errorRate.append(err)

    errorRate = np.asarray(errorRate)
    np.savetxt('./Results/BBB_epochs_errorRate.csv', errorRate, delimiter=",")
    #plt.plot(range(TRAIN_EPOCHS), errorRate, c='royalblue', label='Bayes BackProp')
    #plt.legend()
    #plt.tight_layout()
    #plt.savefig('./Results/MNIST_EPOCHS.png')
    #torch.save(mnist.net.state_dict(), './Models/BBB_MNIST.pth')

# Scalar Mixture vs Gaussian
def MixtureVsGaussianAnalyis():

    #Hyperparameter setting
    BATCH_SIZE = 125
    TEST_BATCH_SIZE = 1000
    CLASSES = 10
    SAMPLES = 2
    TRAIN_EPOCHS = 600
    PI = 0.5
    SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
    SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])
    INPUT_SIZE = 28*28
    LAYERS = np.array([[400,400],[800,800],[1200,1200]]) #Possible layer configuration
    reading = []

    for l in range(LAYERS.shape[0]):
        layer = np.asarray(LAYERS[l])
        print("Network architecture: ",layer)

        #one with scalar mixture gaussian prior
        mnist = MNIST(BATCH_SIZE = BATCH_SIZE,\
                TEST_BATCH_SIZE = TEST_BATCH_SIZE,\
                CLASSES = CLASSES,\
                TRAIN_EPOCHS = TRAIN_EPOCHS,\
                SAMPLES = SAMPLES,\
                hasScalarMixturePrior = True,\
                PI = PI,\
                SIGMA_1 = SIGMA_1,\
                SIGMA_2 = SIGMA_2,\
                INPUT_SIZE = INPUT_SIZE,\
                LAYERS = layer,\
                ACTIVATION_FUNCTIONS = np.array(['relu','relu','softmax']))

        #one with simple gaussian prior
        mnistGaussian = MNIST(BATCH_SIZE = BATCH_SIZE,\
                    TEST_BATCH_SIZE = TEST_BATCH_SIZE,\
                    CLASSES = CLASSES,\
                    TRAIN_EPOCHS = TRAIN_EPOCHS,\
                    SAMPLES = SAMPLES,\
                    hasScalarMixturePrior = False,\
                    PI = PI,\
                    SIGMA_1 = SIGMA_1,\
                    SIGMA_2 = SIGMA_2,\
                    INPUT_SIZE = INPUT_SIZE,\
                    LAYERS = layer,\
                    ACTIVATION_FUNCTIONS = np.array(['relu','relu','softmax']))
        
        for _ in tqdm(range(TRAIN_EPOCHS)):
            mnist.train()
            mnistGaussian.train()
        print("Testing begins!")
        reading.append([layer[0],mnist.test(),mnistGaussian.test()])
    
    reading = np.asarray(reading)
    np.savetxt('./Results/BBB_scalarVsGaussian.csv',reading,delimiter=",")

# Different values of sample, pi, sigma 1 and sigma 2
def HyperparameterAnalysis():
    
    #hyper parameter declaration
    BATCH_SIZE = 1000
    TEST_BATCH_SIZE = 1000
    CLASSES = 10
    TRAIN_EPOCHS = 100
    SAMPLES = np.array([1,2,5,10]) #possible values of sample size
    PI = np.array([0.25,0.5,0.75]) #possible values of pi
    SIGMA_1 = np.array([0,1,2]) #possible values of sigma1 
    SIGMA_2 = np.array([6,7,8]) #possible values of sigma2
    INPUT_SIZE = 28*28
    LAYERS = np.array([400,400])
    LR = [1e-4, 1e-2, 1e-3, 1e-5]

    errorRate = []
    for sample in range(SAMPLES.size):
        for pi in range(PI.size):
            for sigma1 in range(SIGMA_1.size):
                for sigma2 in range(SIGMA_2.size):
                    for lr in range(len(LR)):

                        mnist = MNIST(BATCH_SIZE = BATCH_SIZE,\
                                    TEST_BATCH_SIZE = TEST_BATCH_SIZE,\
                                    CLASSES = CLASSES,\
                                    TRAIN_EPOCHS = TRAIN_EPOCHS,\
                                    SAMPLES = SAMPLES[sample],\
                                    hasScalarMixturePrior = True,\
                                    PI = PI[pi],\
                                    SIGMA_1 = torch.cuda.FloatTensor([math.exp(-SIGMA_1[sigma1])]),\
                                    SIGMA_2 = torch.cuda.FloatTensor([math.exp(-SIGMA_2[sigma2])]),\
                                    INPUT_SIZE = INPUT_SIZE,\
                                    LAYERS = LAYERS,\
                                    ACTIVATION_FUNCTIONS = np.array(['relu','relu','softmax']),\
                                    LR = LR[lr])

                        print(SAMPLES[sample],PI[pi],SIGMA_1[sigma1],SIGMA_2[sigma2],LR[lr])

                        for _ in tqdm(range(TRAIN_EPOCHS)):
                            mnist.train()
                        
                        errorRate.append([SAMPLES[sample],PI[pi],SIGMA_1[sigma1],SIGMA_2[sigma2],LR[lr],mnist.test()])
                        errorRate = np.asarray(errorRate)
                        np.savetxt('./Results/BBB_hyperparameters.csv', errorRate, delimiter=",")

if __name__ == '__main__':
    multipleEpochAnalyis()
    #MixtureVsGaussianAnalyis()
    #HyperparameterAnalysis()