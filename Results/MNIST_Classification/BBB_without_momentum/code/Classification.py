import math

import torch
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

torch.manual_seed(0)

from SGD import *
from BayesBackpropagation import *
from data.dataset import MNISTDataset
from data.parser import parse_mnist
from data.transforms import MNISTTransform

class MNIST(object):
    def __init__(self, BATCH_SIZE, TEST_BATCH_SIZE, CLASSES, TRAIN_EPOCHS, SAMPLES, hasScalarMixturePrior, PI, SIGMA_1,
                 SIGMA_2, INPUT_SIZE, LAYERS, ACTIVATION_FUNCTIONS, LR, DATAMODE='mlp', GOOGLE_INIT=False,
                 train_loader=None, valid_loader=None, test_loader=None, momentum=0):
        if train_loader is None or valid_loader is None or test_loader is None:
            # Prepare data
            if DATAMODE == 'mlp':
                train_data, train_label, valid_data, valid_label, test_data, test_label = parse_mnist(2)
            elif DATAMODE == 'cnn':
                train_data, train_label, valid_data, valid_label, test_data, test_label = parse_mnist(4)
            else:
                raise ValueError('Unsupported mode')

            train_dataset = MNISTDataset(train_data, train_label, transform=MNISTTransform())
            valid_dataset = MNISTDataset(valid_data, valid_label, transform=MNISTTransform())
            test_dataset = MNISTDataset(test_data, test_label, transform=MNISTTransform())
            self.train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, **LOADER_KWARGS)
            self.valid_loader = DataLoader(valid_dataset, TEST_BATCH_SIZE, shuffle=False, **LOADER_KWARGS)
            self.test_loader = DataLoader(test_dataset, TEST_BATCH_SIZE, shuffle=False, **LOADER_KWARGS)
        else:
            self.train_loader = train_loader
            self.valid_loader = valid_loader
            self.test_loader = test_loader

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
        self.net = BayesianNetwork(inputSize=INPUT_SIZE,
                                   CLASSES=CLASSES,
                                   layers=LAYERS,
                                   activations=ACTIVATION_FUNCTIONS,
                                   SAMPLES=SAMPLES,
                                   BATCH_SIZE=BATCH_SIZE,
                                   NUM_BATCHES=self.NUM_BATCHES,
                                   hasScalarMixturePrior=hasScalarMixturePrior,
                                   PI=PI,
                                   SIGMA_1=SIGMA_1,
                                   SIGMA_2=SIGMA_2,
                                   GOOGLE_INIT=GOOGLE_INIT).to(DEVICE)

        # Optimizer declaration
        self.optimizer = optim.SGD(self.net.parameters(), lr=LR, momentum=momentum)  # self.optimizer = optim.Adam(self.net.parameters())


    # Define the training step for MNIST data set
    def train(self):
        loss = 0.
        for batch_idx, (input, target) in enumerate(self.train_loader):
            input, target = input.to(DEVICE), target.to(DEVICE)
            self.net.zero_grad()

            loss = self.net.BBB_loss(input, target)
            if math.isnan(loss):
                print('Training failed. Loss is out of bounds. Try to adapt LR.')
                sys.exit(0)

            loss.backward()
            self.optimizer.step()
        return loss

    # Testing the ensemble
    def test(self, valid=True, TEST_SAMPLES=1):
        data_loader = self.valid_loader if valid else self.test_loader
        correct = 0
        with torch.no_grad():
            for input, target in data_loader:
                input, target = input.to(DEVICE), target.to(DEVICE)

                outputs = torch.zeros(TEST_SAMPLES, self.TEST_BATCH_SIZE, self.CLASSES).to(DEVICE)
                if TEST_SAMPLES == 1:
                    output = self.net.forward(input, infer=True)
                for i in range(TEST_SAMPLES):
                    outputs[i] = self.net.forward(input, infer=False)
                    output = outputs.mean(0)

                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / self.TEST_SIZE

        return round(100 * (1 - accuracy), 3)  # Error

# Different values of sample, pi, sigma 1 and sigma 2
def HyperparameterAnalysis():
    import sys
    #samples = 1

    # hyper parameter declaration
    BATCH_SIZE = 125
    TEST_BATCH_SIZE = 1000
    CLASSES = 10
    TRAIN_EPOCHS = 600
    SAMPLES = np.array([1, 2])  # possible values of sample size
    PI = np.array([0.75])  # possible values of pi
    SIGMA_1 = np.array([1])  # possible values of sigma1
    SIGMA_2 = np.array([8])  # possible values of sigma2
    INPUT_SIZE = 28 * 28
    LAYERS = np.array([400, 400])
    #LR = [1e-4]
    #momentum = 0.95
    LR = [1e-3]
    momentum = 0
    GOOGLE_INIT=True
    SCALARS=[6.0, 5.5, 5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]

    errorRate = []
    for sample in range(SAMPLES.size):
        samples=SAMPLES[sample]
        for pi in range(PI.size):
                for sigma1 in range(SIGMA_1.size):
                    for sigma2 in range(SIGMA_2.size):
                        for lr in range(len(LR)):
                            for scalars in range(len(SCALARS)):
                                scalar = 10. ** (-SCALARS[scalars])
                                deviceScalar = torch.FloatTensor([scalar]).to(DEVICE)

                                mnist = MNIST(BATCH_SIZE=BATCH_SIZE,
                                              TEST_BATCH_SIZE=TEST_BATCH_SIZE,
                                              CLASSES=CLASSES,
                                              TRAIN_EPOCHS=TRAIN_EPOCHS,
                                              SAMPLES=samples,
                                              hasScalarMixturePrior=True,
                                              PI=PI[pi],
                                              SIGMA_1=torch.cuda.FloatTensor([math.exp(-SIGMA_1[sigma1])]),
                                              SIGMA_2=torch.cuda.FloatTensor([math.exp(-SIGMA_2[sigma2])]),
                                              INPUT_SIZE=INPUT_SIZE,
                                              LAYERS=LAYERS,
                                              GOOGLE_INIT=GOOGLE_INIT,
                                              ACTIVATION_FUNCTIONS=np.array(['relu', 'relu', 'softmax']),
                                              LR=LR[lr],
                                              momentum=momentum)

                                print(samples, PI[pi], SIGMA_1[sigma1], SIGMA_2[sigma2], LR[lr], momentum, GOOGLE_INIT, scalar)

                                for epoch in tqdm(range(TRAIN_EPOCHS)):
                                    loss = mnist.train(scalar=deviceScalar)
                                    acc = mnist.test()
                                    testAcc = mnist.test(valid=False)
                                    errorRate.append(
                                        [samples, PI[pi], SIGMA_1[sigma1], SIGMA_2[sigma2], LR[lr], epoch + 1, scalar, acc, testAcc])
                                    np.savetxt('./Results/BBB_hyperparameters_google_lr1e-3_scalars.csv', errorRate,
                                               delimiter=",")
                                    print(acc, float(loss))

                                sub = str(samples) + '_' + str(PI[pi]) + '_' +str(SIGMA_1[sigma1]) + '_' +str(SIGMA_2[sigma2]) + '_' +str(LR[lr]) + '_scalar' +str(scalar)
                                torch.save(mnist.net.state_dict(), './Models/BBB_hyperparameters_google_samples' + sub + '.pth')

def classify(MODEL, HIDDEN_UNITS, TRAIN_EPOCHS, DATASET, BATCH_SIZE=125, TEST_BATCH_SIZE=1000, momentum = 0.95):
    """Set model"""
    if DATASET == 'mnist':
        train_data, train_label, valid_data, valid_label, test_data, test_label = parse_mnist(2)

        train_dataset = MNISTDataset(train_data, train_label, transform=MNISTTransform())
        valid_dataset = MNISTDataset(valid_data, valid_label, transform=MNISTTransform())
        test_dataset = MNISTDataset(test_data, test_label, transform=MNISTTransform())
        train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, **LOADER_KWARGS)
        valid_loader = DataLoader(valid_dataset, TEST_BATCH_SIZE, shuffle=False, **LOADER_KWARGS)
        test_loader = DataLoader(test_dataset, TEST_BATCH_SIZE, shuffle=False, **LOADER_KWARGS)

        INPUT_SIZE = 28 * 28
        CLASSES = 10
    else:
        raise ValueError('Valid params: DATASET=mnist')

    if MODEL== 'bbb':
        LR = 1e-4

        # Define the used hyperparameters
        if HIDDEN_UNITS==400:
            SAMPLES = 1
            PI = 0.25
            SIGMA_1 = 1.
            SIGMA_2 = 8.
            GOOGLE_INIT = True
        elif HIDDEN_UNITS==800:
            SAMPLES = 1
            PI = 0.75
            SIGMA_1 = 1.
            SIGMA_2 = 6.
            GOOGLE_INIT = True
        elif HIDDEN_UNITS==1200:
            SAMPLES = 1
            PI = 0.75
            SIGMA_1 = 0.
            SIGMA_2 = 7.
            GOOGLE_INIT = True
        else:
            raise ValueError('Valid params: HIDDEN_UNITS=400|800|1200')
        LAYERS = np.array([HIDDEN_UNITS, HIDDEN_UNITS])

        if momentum == 0:
            LR = 1e-3

        # errorRate = []  # to store error rates at different epochs

        print('HIDDEN_UNITS', 'TRAIN_EPOCHS', 'SAMPLES', 'PI', 'SIGMA_1', 'SIGMA_2', 'LR', 'GOOGLE_INIT')
        print(HIDDEN_UNITS, TRAIN_EPOCHS, SAMPLES, PI, SIGMA_1, SIGMA_2, LR, GOOGLE_INIT)

        mnist = MNIST(BATCH_SIZE=BATCH_SIZE,
                      TEST_BATCH_SIZE=TEST_BATCH_SIZE,
                      CLASSES=CLASSES,
                      TRAIN_EPOCHS=TRAIN_EPOCHS,
                      SAMPLES=SAMPLES,
                      hasScalarMixturePrior=True,
                      PI=PI,
                      SIGMA_1=torch.FloatTensor([math.exp(-SIGMA_1)]).to(DEVICE),
                      SIGMA_2=torch.FloatTensor([math.exp(-SIGMA_2)]).to(DEVICE),
                      INPUT_SIZE=INPUT_SIZE,
                      LAYERS=LAYERS,
                      ACTIVATION_FUNCTIONS=np.array(['relu', 'relu', 'softmax']),
                      LR=LR,
                      GOOGLE_INIT=GOOGLE_INIT,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      valid_loader=valid_loader,
                      momentum=momentum)

        train_losses = np.zeros(TRAIN_EPOCHS)
        valid_errs = np.zeros(TRAIN_EPOCHS)
        test_errs = np.zeros(TRAIN_EPOCHS)

        print('epoch', 'validErr', 'testErr', 'loss')

        for epoch in tqdm(range(TRAIN_EPOCHS)):
            loss = mnist.train()
            validErr, testErr = mnist.test(valid=True), mnist.test(valid=False)

            print(epoch + 1, validErr, testErr, float(loss))

            valid_errs[epoch] = validErr
            test_errs[epoch] = testErr
            train_losses[epoch] = float(loss)

        # Save results
        path = 'Models/BBB_MNIST_' + str(HIDDEN_UNITS) + '_' + str(momentum) + '_1samples_' + str(TRAIN_EPOCHS) + 'epochs'
        wr = csv.writer(open(path + '.csv', 'w'), delimiter=',', lineterminator='\n')
        wr.writerow(['epoch', 'valid_acc', 'test_acc', 'train_losses'])
        for i in range(TRAIN_EPOCHS):
            wr.writerow((i + 1, valid_errs[i], test_errs[i], train_losses[i]))

        torch.save(mnist.net.state_dict(), path + '.pth')
    elif MODEL=='dropout' or MODEL=='mlp':
        hyper = SGD_Hyper()
        hyper.hidden_units = HIDDEN_UNITS
        hyper.max_epoch = TRAIN_EPOCHS
        hyper.mode = MODEL

        # Train and save results
        SGD_run(hyper, train_loader=train_loader, test_loader=test_loader, valid_loader=valid_loader)


if __name__ == '__main__':
    classify(MODEL='bbb', HIDDEN_UNITS=400, TRAIN_EPOCHS=600, DATASET='mnist')
    classify(MODEL='bbb', HIDDEN_UNITS=800, TRAIN_EPOCHS=600, DATASET='mnist')
    classify(MODEL='bbb', HIDDEN_UNITS=1200, TRAIN_EPOCHS=600, DATASET='mnist')