# Deep Bayesian Bandits Library

This library corresponds to the *[Deep Bayesian Bandits Showdown: An Empirical
Comparison of Bayesian Deep Networks for Thompson
Sampling](https://arxiv.org/abs/1802.09127)* paper, published in
[ICLR](https://iclr.cc/) 2018. We provide a benchmark to test decision-making
algorithms for contextual-bandits. In particular, the current library implements
a variety of algorithms (many of them based on approximate Bayesian Neural
Networks and Thompson sampling), and a number of real and syntethic data
problems exhibiting a diverse set of properties.

It is a Python library that uses [TensorFlow](https://www.tensorflow.org/).

We encourage contributors to add new approximate Bayesian Neural Networks or,
more generally, contextual bandits algorithms to the library. Also, we would
like to extend the data sources over time, so we warmly encourage contributions
in this front too!

Please, use the following when citing the code or the paper:

```
@article{riquelme2018deep, title={Deep Bayesian Bandits Showdown: An Empirical
Comparison of Bayesian Deep Networks for Thompson Sampling},
author={Riquelme, Carlos and Tucker, George and Snoek, Jasper},
journal={International Conference on Learning Representations, ICLR.}, year={2018}}
```

**Contact**. This repository is maintained by [Carlos Riquelme](http://rikel.me) ([rikel](https://github.com/rikel)). Feel free to reach out directly at [rikel@google.com](mailto:rikel@google.com) with any questions or comments.


We first briefly introduce contextual bandits, Thompson sampling, enumerate the
implemented algorithms, and the available data sources. Then, we provide a
simple complete example illustrating how to use the library.

## Contextual Bandits

Contextual bandits are a rich decision-making framework where an algorithm has
to choose among a set of *k* actions at every time step *t*, after observing
a context (or side-information) denoted by *X<sub>t</sub>*. The general pseudocode for
the process if we use algorithm **A** is as follows:

```
At time t = 1, ..., T:
  1. Observe new context: X_t
  2. Choose action: a_t = A.action(X_t)
  3. Observe reward: r_t
  4. Update internal state of the algorithm: A.update((X_t, a_t, r_t))
```

The goal is to maximize the total sum of rewards: &sum;<sub>t</sub> r<sub>t</sub>

For example, each *X<sub>t</sub>* could encode the properties of a specific user (and
the time or day), and we may have to choose an ad, discount coupon, treatment,
hyper-parameters, or version of a website to show or provide to the user.
Hopefully, over time, we will learn how to match each type of user to the most
beneficial personalized action under some metric (the reward).

## Thompson Sampling

Thompson Sampling is a meta-algorithm that chooses an action for the contextual
bandit in a statistically efficient manner, simultaneously finding the best arm
while attempting to incur low cost. Informally speaking, we assume the expected
reward is given by some function
**E**[r<sub>t</sub> | X<sub>t</sub>, a<sub>t</sub>] = f(X<sub>t</sub>, a<sub>t</sub>).
Unfortunately, function **f** is unknown, as otherwise we could just choose the
action with highest expected value:
a<sub>t</sub><sup>*</sup> = arg max<sub>i</sub> f(X<sub>t</sub>, a<sub>t</sub>).

The idea behind Thompson Sampling is based on keeping a posterior distribution
&pi;<sub>t</sub> over functions in some family f &isin; F after observing the first
*t-1* datapoints. Then, at time *t*, we sample one potential explanation of
the underlying process: f<sub>t</sub> &sim; &pi;<sub>t</sub>, and act optimally (i.e., greedily)
*according to f<sub>t</sub>*. In other words, we choose
a<sub>t</sub> = arg max<sub>i</sub> f<sub>t</sub>(X<sub>t</sub>, a<sub>i</sub>).
Finally, we update our posterior distribution with the new collected
datapoint (X<sub>t</sub>, a<sub>t</sub>, r<sub>t</sub>).

The main issue is that keeping an updated posterior &pi;<sub>t</sub> (or, even,
sampling from it) is often intractable for highly parameterized models like deep
neural networks. The algorithms we list in the next section provide tractable
*approximations* that can be used in combination with Thompson Sampling to solve
the contextual bandit problem.

## Algorithm

4.  **Stochastic Variational Inference**, Bayes by Backpropagation. We implement
    a Bayesian neural network by modeling each individual weight posterior as a
    univariate Gaussian distribution: w<sub>ij</sub> &sim; N(&mu;<sub>ij</sub>, &sigma;<sub>ij</sub><sup>2</sup>).
    Thompson sampling then samples a network at each time step
    by sampling each weight independently. The variational approach consists in
    maximizing a proxy for maximum likelihood of the observed data, the ELBO or
    variational lower bound, to fit the values of &mu;<sub>ij</sub>, &sigma;<sub>ij</sub><sup>2</sup>
    for every *i, j*.

    See [Weight Uncertainty in Neural
    Networks](https://arxiv.org/abs/1505.05424).

    The BNN algorithm is implemented in *variational_neural_bandit_model.py*,
    and it is used together with *PosteriorBNNSampling* (defined in
    *posterior_bnn_sampling.py*) by calling:

    ```
        bbb = PosteriorBNNSampling('myBBB', my_hparams, 'Variational')
    ```

## Usage: Basic Example

This library requires Tensorflow, Numpy, and Pandas.

The file *example_main.py* provides a complete example on how to use the
library. We run the code:

```
    python example_main.py
```

**Do not forget to** configure the routes to the data files at the top of *example_main.py*.

For example, we can run the Mushroom bandit for 2000 contexts on a few
algorithms as follows:

```
  # Problem parameters
  num_contexts = 2000

  # Choose data source among:
  # {linear, sparse_linear, mushroom, financial, jester,
  #  statlog, adult, covertype, census, wheel}
  data_type = 'mushroom'

  # Create dataset
  sampled_vals = sample_data(data_type, num_contexts)
  dataset, opt_rewards, opt_actions, num_actions, context_dim = sampled_vals

  # Define hyperparameters and algorithms
  hparams_linear = tf.contrib.training.HParams(num_actions=num_actions,
                                               context_dim=context_dim,
                                               a0=6,
                                               b0=6,
                                               lambda_prior=0.25,
                                               initial_pulls=2)

  hparams_dropout = tf.contrib.training.HParams(num_actions=num_actions,
                                                context_dim=context_dim,
                                                init_scale=0.3,
                                                activation=tf.nn.relu,
                                                layer_sizes=[50],
                                                batch_size=512,
                                                activate_decay=True,
                                                initial_lr=0.1,
                                                max_grad_norm=5.0,
                                                show_training=False,
                                                freq_summary=1000,
                                                buffer_s=-1,
                                                initial_pulls=2,
                                                optimizer='RMS',
                                                reset_lr=True,
                                                lr_decay_rate=0.5,
                                                training_freq=50,
                                                training_epochs=100,
                                                keep_prob=0.80,
                                                use_dropout=True)

  ### Create hyper-parameter configurations for other algorithms
    [...]

  algos = [
      UniformSampling('Uniform Sampling', hparams),
      PosteriorBNNSampling('Dropout', hparams_dropout, 'RMSProp'),
      PosteriorBNNSampling('BBB', hparams_bbb, 'Variational'),
      NeuralLinearPosteriorSampling('NeuralLinear', hparams_nlinear),
      LinearFullPosteriorSampling('LinFullPost', hparams_linear),
      BootstrappedBNNSampling('BootRMS', hparams_boot),
      ParameterNoiseSampling('ParamNoise', hparams_pnoise),
  ]

  # Run contextual bandit problem
  t_init = time.time()
  results = run_contextual_bandit(context_dim, num_actions, dataset, algos)
  _, h_rewards = results

  # Display results
  display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, data_type)

```

The previous code leads to final results that look like:

```
---------------------------------------------------
---------------------------------------------------
mushroom bandit completed after 69.8401839733 seconds.
---------------------------------------------------
  0) LinFullPost         |               total reward =     4365.0.
  1) NeuralLinear        |               total reward =     4110.0.
  2) Dropout             |               total reward =     3430.0.
  3) ParamNoise          |               total reward =     3270.0.
  4) BootRMS             |               total reward =     3050.0.
  5) BBB                 |               total reward =     2505.0.
  6) Uniform Sampling    |               total reward =    -4930.0.
---------------------------------------------------
Optimal total reward = 5235.
Frequency of optimal actions (action, frequency):
[[0, 953], [1, 1047]]
---------------------------------------------------
---------------------------------------------------
```
