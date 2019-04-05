# "Weight Uncertainty in Neural Networks"

This project aims to reproduce the results of  Blundell et al. in their paper "Weight Uncertainty in Neural Networks" from 2015.
This paper presents Bayesian neural networks that only require a minor modification to the backpropagation algorithm, which they call "Bayes by Backprop" (BBB).
We also perform a number of extensions further to the original publication.

## Experiments
You find the performed experiments in individual folders, alongside our results in the directory `Results/`.

We performed the following experiments:
- _Image classification_ through fully-connected networks (FCN) in `FCN_Image_Classification/`,
- _Regression_ on a toy example in `Regression/`,
- Reinforcement learning in _contextual bandits_ in `Contextual Bandit`,
- (Extension) _Pokémon_ type classification from colour in `Pokemon/`,
- (Extension) A _PyTorch BBB framework_ to apply BBB to any network architecture, studied for CNNs and DensetNets, in `BayesCNN/`.

The most notable contribution is our BBB PyTorch framework. As far we are concerned, there has thus far not existed a comparably versatile implementation of BBB.

### Image classification through FCNs
There are two main scripts in the `FCN_Image_Classification/` folder, `SGD.py` and `BBB.py`. The latter runs image classification through BBB, whereas `SGD.py` for regular FCNs, with or without dropout.

The BBB experiments are called as follows:
```python
python3 BBB.py [hidden units per layer] [mnist|fmnist|cifar10]
```
The SGD experiments are executed in the same fashion:
```python
python3 SGD.py [hidden units per layer] [mnist|fmnist|cifar10] [mlp|dropout]
```
Both scripts run for 600 epochs, then save a report and the generated model in `Results/`.

We also provide a script for weight pruning, that is run on the resulting model as follows:
```python
python3 WeightPruning.py [hidden units per layer] [path to model]
```
Again, this script stores its results in `Results/`. The results consist of pruned models and plots, as in the original paper.
The models can then be further tested for their classification errors.
For the case of BBB, you can just uncomment the last few lines of `BBB.py` script.

### Regression
### Contextual Bandits
### Pokémon Type Classification
### BBB framework

## References
- C. Blundell, J. Cornebise, K. Kavukcuoglu and D. Wierstra, [‘Weight uncertainty in
neural networks’](https://arxiv.org/abs/1505.05424), in Proceedings of the 32nd International Conference on International Conference on Machine Learning - Volume 37, ser. ICML’15, 2015, pp. 1613–1622.
