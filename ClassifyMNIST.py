from MNIST import *
import numpy as np

# Multiple epochs
def multipleEpochAnalyis():
    BATCH_SIZE = 100
    TEST_BATCH_SIZE = 5
    CLASSES = 10
    TRAIN_EPOCHS = 600
    SAMPLES = 2
    TEST_SAMPLES = 10
    PI = 0.5
    SIGMA_1 = torch.FloatTensor([math.exp(-0)])
    SIGMA_2 = torch.FloatTensor([math.exp(-6)])
    INPUT_SIZE = 28*28
    LAYERS = np.array([1200,1200])

    errorRate = []
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
                ACTIVATION_FUNCTIONS = np.array(['relu','relu','softmax']))

    for epoch in range(TRAIN_EPOCHS):
        mnist.train()
        #accuracy.append([epoch,mnist.test()])
        errorRate.append(1-mnist.test())

    errorRate = np.asarray(errorRate)
    np.savetxt('./Results/BBB_epochs_errorRate.csv',errorRate,delimiter=",")

    plt.plot(range(TRAIN_EPOCHS), errorRate, c='royalblue', label='Bayes BackProp')
    plt.legend()
    plt.tight_layout()
    plt.savefig('./Results/MNIST_EPOCHS.png')

# Scalar Mixture vs Gaussian
def MixtureVsGaussianAnalyis():
    BATCH_SIZE = 100
    TEST_BATCH_SIZE = 5
    CLASSES = 10
    SAMPLES = 2
    TRAIN_EPOCHS = 600
    TEST_SAMPLES = 10
    PI = 0.5
    SIGMA_1 = torch.FloatTensor([math.exp(-0)])
    SIGMA_2 = torch.FloatTensor([math.exp(-6)])
    INPUT_SIZE = 28*28
    LAYERS = np.array([[400,400],[800,800],[1200,1200]])
    reading = []

    for l in range(LAYERS.shape[0]):
        layer = np.asarray(LAYERS[l])
        print("Network architecture: ",layer)
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
        
        for epoch in range(TRAIN_EPOCHS):
            mnist.train()
            mnistGaussian.train()
        print("Testing begins!")
        reading.append([layer[0],mnist.test(),mnistGaussian.test()])
    
    reading = np.asarray(reading)
    np.savetxt('./Results/BBB_scalarVsGaussian.csv',reading,delimiter=",")

# Different values of sample, pi, sigma 1 and sigma 2
def HyperparameterAnalysis():
    BATCH_SIZE = 100
    TEST_BATCH_SIZE = 5
    CLASSES = 10
    TRAIN_EPOCHS = 600
    SAMPLES = np.array([1])
    TEST_SAMPLES = 10
    PI = np.array([0.25,0.5,0.75])
    SIGMA_1 = np.array([0,1,2])
    SIGMA_2 = np.array([6,7,8])
    INPUT_SIZE = 28*28
    LAYERS = np.array([400,400])

    errorRate = []
    for sample in range(SAMPLES.size):
        for pi in range(PI.size):
            for sigma1 in range(SIGMA_1.size):
                for sigma2 in range(SIGMA_2.size):

                    mnist = MNIST(BATCH_SIZE = BATCH_SIZE,\
                                TEST_BATCH_SIZE = TEST_BATCH_SIZE,\
                                CLASSES = CLASSES,\
                                TRAIN_EPOCHS = TRAIN_EPOCHS,\
                                SAMPLES = SAMPLES[sample],\
                                TEST_SAMPLES = TEST_SAMPLES,\
                                hasScalarMixturePrior = True,\
                                PI = PI[pi],\
                                SIGMA_1 = torch.FloatTensor([math.exp(-SIGMA_1[sigma1])]),\
                                SIGMA_2 = torch.FloatTensor([math.exp(-SIGMA_2[sigma2])]),\
                                INPUT_SIZE = INPUT_SIZE,\
                                LAYERS = LAYERS,\
                                ACTIVATION_FUNCTIONS = np.array(['relu','relu','softmax']))

                    for epoch in range(TRAIN_EPOCHS):
                        mnist.train()
                    
                    errorRate.append([SAMPLES[sample],PI[pi],SIGMA_1[sigma1],SIGMA_2[sigma2],1-mnist.test()])

    errorRate = np.asarray(errorRate)
    np.savetxt('./Results/BBB_hyperparameters.csv',errorRate,delimiter=",")

multipleEpochAnalyis()
MixtureVsGaussianAnalyis()
HyperparameterAnalysis()