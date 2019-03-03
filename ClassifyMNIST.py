from MNIST import *
import numpy as np
from tqdm import tqdm

# Multiple epochs
def multipleEpochAnalyis():
    torch.manual_seed(0)  # for reproducibility

    #Hyperparameter declaration
    BATCH_SIZE = 125
    TEST_BATCH_SIZE = 1000
    CLASSES = 10
    TRAIN_EPOCHS = 100
    SAMPLES = 1
    TEST_SAMPLES = 1
    PI = 0.25
    #SIGMA_1 = torch.FloatTensor([math.exp(-0)])
    #SIGMA_2 = torch.FloatTensor([math.exp(-6)])
    #if torch.cuda.is_available():
    SIGMA_1 = torch.cuda.FloatTensor([0.75]) # torch.cuda.FloatTensor([math.exp(-0)])
    SIGMA_2 = torch.cuda.FloatTensor([0.1]) # torch.cuda.FloatTensor([math.exp(-6)])
    INPUT_SIZE = 28*28
    LAYERS = np.array([400,400])

    errorRate = [] #to store error rates at different epochs

    #Declare object of class MNIST declared in MNIST.py
    mnist = MNIST(BATCH_SIZE = BATCH_SIZE,\
                TEST_BATCH_SIZE = TEST_BATCH_SIZE,\
                CLASSES = CLASSES,\
                TRAIN_EPOCHS = TRAIN_EPOCHS,\
                SAMPLES = SAMPLES,\
                TEST_SAMPLES = TEST_SAMPLES,\
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
        #acc = 1-mnist.test()
        print(mnist.test(), float(loss))
        #errorRate.append(acc) # 1-accuracy

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
    TEST_SAMPLES = 10
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
                TEST_SAMPLES = TEST_SAMPLES,\
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
                    TEST_SAMPLES = TEST_SAMPLES,\
                    hasScalarMixturePrior = False,\
                    PI = PI,\
                    SIGMA_1 = SIGMA_1,\
                    SIGMA_2 = SIGMA_2,\
                    INPUT_SIZE = INPUT_SIZE,\
                    LAYERS = layer,\
                    ACTIVATION_FUNCTIONS = np.array(['relu','relu','softmax']))
        
        for epoch in tqdm(range(TRAIN_EPOCHS)):
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
    TEST_SAMPLES = 10
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
                                    TEST_SAMPLES = TEST_SAMPLES,\
                                    hasScalarMixturePrior = True,\
                                    PI = PI[pi],\
                                    SIGMA_1 = torch.cuda.FloatTensor([math.exp(-SIGMA_1[sigma1])]),\
                                    SIGMA_2 = torch.cuda.FloatTensor([math.exp(-SIGMA_2[sigma2])]),\
                                    INPUT_SIZE = INPUT_SIZE,\
                                    LAYERS = LAYERS,\
                                    ACTIVATION_FUNCTIONS = np.array(['relu','relu','softmax']),\
                                    LR = LR[lr])

                        print(SAMPLES[sample],PI[pi],SIGMA_1[sigma1],SIGMA_2[sigma2],LR[lr])

                        for epoch in tqdm(range(TRAIN_EPOCHS)):
                            mnist.train()
                        
                        errorRate.append([SAMPLES[sample],PI[pi],SIGMA_1[sigma1],SIGMA_2[sigma2],LR[lr],1-mnist.test()])

    errorRate = np.asarray(errorRate)
    np.savetxt('./Results/BBB_hyperparameters.csv',errorRate,delimiter=",")

multipleEpochAnalyis()
#MixtureVsGaussianAnalyis()
#HyperparameterAnalysis()