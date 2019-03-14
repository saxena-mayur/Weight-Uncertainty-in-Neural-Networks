# This is the README

We want to - reparameterize by sigma squared, not sigma.

The paper BayesCNN implements the local reparameterization trick but still sample weights, so it is still inefficient.

Either we implement BayesCNN with weight sampling, or we can use the prior from the 'local reparameterization trick' paper and the approximate kl-div.

compute approximate kl-div for mixture of gaussian since sigma2< < sigma1
No we will not do this.

We first implement BayesCNN without reparameterization trick.
Then,

We will implement scale-invariant log-uniform prior.

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

What to Do:
- Visualise distribution of weights and standard deviations
- Try on DenseNets
