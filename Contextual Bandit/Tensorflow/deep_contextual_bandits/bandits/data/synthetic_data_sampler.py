# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Several functions to sample contextual data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def sample_contextual_data(num_contexts, dim_context, num_actions, sigma):
  """Samples independent Gaussian data.

  There is nothing to learn here as the rewards do not depend on the context.

  Args:
    num_contexts: Number of contexts to sample.
    dim_context: Dimension of the contexts.
    num_actions: Number of arms for the multi-armed bandit.
    sigma: Standard deviation of the independent Gaussian samples.

  Returns:
    data: A [num_contexts, dim_context + num_actions] numpy array with the data.
  """
  size_data = [num_contexts, dim_context + num_actions]
  return np.random.normal(scale=sigma, size=size_data)


def sample_linear_data(num_contexts, dim_context, num_actions, sigma=0.0):
  """Samples data from linearly parameterized arms.

  The reward for context X and arm j is given by X^T beta_j, for some latent
  set of parameters {beta_j : j = 1, ..., k}. The beta's are sampled uniformly
  at random, the contexts are Gaussian, and sigma-noise is added to the rewards.

  Args:
    num_contexts: Number of contexts to sample.
    dim_context: Dimension of the contexts.
    num_actions: Number of arms for the multi-armed bandit.
    sigma: Standard deviation of the additive noise. Set to zero for no noise.

  Returns:
    data: A [n, d+k] numpy array with the data.
    betas: Latent parameters that determine expected reward for each arm.
    opt: (optimal_rewards, optimal_actions) for all contexts.
  """

  betas = np.random.uniform(-1, 1, (dim_context, num_actions))
  betas /= np.linalg.norm(betas, axis=0)
  contexts = np.random.normal(size=[num_contexts, dim_context])
  rewards = np.dot(contexts, betas)
  opt_actions = np.argmax(rewards, axis=1)
  rewards += np.random.normal(scale=sigma, size=rewards.shape)
  opt_rewards = np.array([rewards[i, act] for i, act in enumerate(opt_actions)])
  return np.hstack((contexts, rewards)), betas, (opt_rewards, opt_actions)


def sample_sparse_linear_data(num_contexts, dim_context, num_actions,
                              sparse_dim, sigma=0.0):
  """Samples data from sparse linearly parameterized arms.

  The reward for context X and arm j is given by X^T beta_j, for some latent
  set of parameters {beta_j : j = 1, ..., k}. The beta's are sampled uniformly
  at random, the contexts are Gaussian, and sigma-noise is added to the rewards.
  Only s components out of d are non-zero for each arm's beta.

  Args:
    num_contexts: Number of contexts to sample.
    dim_context: Dimension of the contexts.
    num_actions: Number of arms for the multi-armed bandit.
    sparse_dim: Dimension of the latent subspace (sparsity pattern dimension).
    sigma: Standard deviation of the additive noise. Set to zero for no noise.

  Returns:
    data: A [num_contexts, dim_context+num_actions] numpy array with the data.
    betas: Latent parameters that determine expected reward for each arm.
    opt: (optimal_rewards, optimal_actions) for all contexts.
  """

  flatten = lambda l: [item for sublist in l for item in sublist]
  sparse_pattern = flatten(
      [[(j, i) for j in np.random.choice(range(dim_context),
                                         sparse_dim,
                                         replace=False)]
       for i in range(num_actions)])
  betas = np.random.uniform(-1, 1, (dim_context, num_actions))
  mask = np.zeros((dim_context, num_actions))
  for elt in sparse_pattern:
    mask[elt] = 1
  betas = np.multiply(betas, mask)
  betas /= np.linalg.norm(betas, axis=0)
  contexts = np.random.normal(size=[num_contexts, dim_context])
  rewards = np.dot(contexts, betas)
  opt_actions = np.argmax(rewards, axis=1)
  rewards += np.random.normal(scale=sigma, size=rewards.shape)
  opt_rewards = np.array([rewards[i, act] for i, act in enumerate(opt_actions)])
  return np.hstack((contexts, rewards)), betas, (opt_rewards, opt_actions)

