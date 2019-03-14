# This is the README



## REPORT

The Aim is to reproduce this graph from the BayesCNN implementation :
put link.

LeNet architecture:

nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(16*5*5),
            nn.Linear(16*5*5, 120), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(120, 84), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(84, 10))

AlexNet architecture:

nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            Flatten(256*2*2),
            nn.Dropout(dropout),
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes))


Densenet architecture: TODO

Results: images on Whatsapp

Dataset: CIFAR 10
prior parameters : PI = 0.75, SIGMA_1 = np.exp(-1), SIGMA_2 = np.exp(-6)
optimizer : ADAM, default parameters (lr = 1e-3)
lr is divided by 3 at steps 33 and 66.
beta: normal
Data Augmentation : only random horizontal flip
Bayes n_samples: 2


comments: using beta as in the blundell paper leads to overfitting: indeed, the beta is significant for only a small number of steps.



# Experiment details:


## Experiment E1

Expected Output: graph of evolution of train accuracy/val accuracy for 100 epochs for the following networks:
AlexNet w/o Dropout, BayesAlexNet w/o Dropout

Configuration
Dataset: CIFAR 10
prior parameters : PI = 0.75, SIGMA_1 = np.exp(-1), SIGMA_2 = np.exp(-6)
optimizer : ADAM, default parameters (lr = 1e-3)
beta: blundell
Data Augmentation : No
Bayes n_samples: 2

Results:
- all accuracies are around 60%, which is very low compared to standard AlexNet implementations
- BayesAlexNet overfits, which suggests that the variances are too low.

Comments:
in the BayesCNN paper, they achieve 80% accuracy with the same AlexNet architecture. Something must be wrong with the hyperparameters.
They use normal initialization, batch-size = 256
edit: fixed, achieve around 74% now, which is standard.

## TODO
- Visualise distribution of weights and standard deviations ?
- Try on DenseNets

## Comments on papers:
The paper BayesCNN implements the local reparameterization trick but still sample weights, so it is still inefficient.

Either we implement BayesCNN with weight sampling, or we can use the prior from the 'local reparameterization trick' paper and the approximate kl-div.

We first implement BayesCNN without reparameterization trick.
Then,

We will implement scale-invariant log-uniform prior.
