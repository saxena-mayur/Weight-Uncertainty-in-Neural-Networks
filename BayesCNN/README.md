# This is the README

We want to - reparameterize by sigma squared, not sigma.

The paper BayesCNN implements the local reparameterization trick but still sample weights, so it is still inefficient.

Either we implement BayesCNN with weight sampling, or we can use the prior from the 'local reparameterization trick' paper and the approximate kl-div.

compute approximate kl-div for mixture of gaussian since sigma2< < sigma1
No we will not do this.

We first implement BayesCNN without reparameterization trick.
Then,

We will implement scale-invariant log-uniform prior.