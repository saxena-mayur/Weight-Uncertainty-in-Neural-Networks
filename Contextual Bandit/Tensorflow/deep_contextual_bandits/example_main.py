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

"""Simple example of contextual bandits simulation.

Code corresponding to:
Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks
for Thompson Sampling, by Carlos Riquelme, George Tucker, and Jasper Snoek.
https://arxiv.org/abs/1802.09127
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from absl import app
from absl import flags
import numpy as np
import os
import tensorflow as tf
import csv

from bandits.core.contextual_bandit import run_contextual_bandit
from bandits.data.data_sampler import sample_mushroom_data
from bandits.algorithms.parameter_noise_sampling import ParameterNoiseSampling
from bandits.algorithms.posterior_bnn_sampling import PosteriorBNNSampling


# Set up your file routes to the data files.
base_route = os.getcwd()
data_route = 'contextual_bandits/datasets'

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)
flags.DEFINE_string('logdir', '/tmp/bandits/', 'Base directory to save output')
flags.DEFINE_string(
    'mushroom_data',
    os.path.join(base_route, data_route, 'mushroom.data'),
    'Directory where Mushroom data is stored.')

def sample_data(data_type, num_contexts=None):
    
  """Sample data from given 'data_type'.

  Args:
    data_type: Dataset from which to sample.
    num_contexts: Number of contexts to sample.

  Returns:
    dataset: Sampled matrix with rows: (context, reward_1, ..., reward_num_act).
    opt_rewards: Vector of expected optimal reward for each context.
    opt_actions: Vector of optimal action for each context.
    num_actions: Number of available actions.
    context_dim: Dimension of each context.
  """

# Create mushroom dataset
  data_type == 'mushroom'
  num_actions = 2
  context_dim = 117
  file_name = FLAGS.mushroom_data
  dataset, opt_mushroom = sample_mushroom_data(file_name, num_contexts)
  opt_rewards, opt_actions = opt_mushroom

  return dataset, opt_rewards, opt_actions, num_actions, context_dim


def display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, name):
  """Displays summary statistics of the performance of each algorithm."""

  print('---------------------------------------------------')
  print('---------------------------------------------------')
  print('{} bandit completed after {} seconds.'.format(
    name, time.time() - t_init))
  print('---------------------------------------------------')

  performance_pairs = []
  regret_pairs = []
  regrets = []
  for j, a in enumerate(algos):
    performance_pairs.append((a.name, np.sum(h_rewards[:, j])))
    regret_pairs.append((a.name, np.cumsum(opt_rewards - h_rewards[:,j])))
    regrets.append(np.cumsum(opt_rewards - h_rewards[:,j]))
  performance_pairs = sorted(performance_pairs,
                             key=lambda elt: elt[1],
                             reverse=True)
  for i, (name, reward) in enumerate(performance_pairs):
    print('{:3}) {:20}| \t \t total reward = {:10}.'.format(i, name, reward))

  print('---------------------------------------------------')
  print('Optimal total reward = {}.'.format(np.sum(opt_rewards)))
  print('Frequency of optimal actions (action, frequency):')
  print([[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)])
  print('---------------------------------------------------')
  print('---------------------------------------------------')

# Export cumulative regrets to CSV
  export_data = zip(*regrets)
  with open('regrets.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
      wr = csv.writer(myfile)
      wr.writerow(("Greedy","Greedy_1","Greedy_5","BBB_Adam_001", "BBB_Adam_0001","BBB_RMS_001","BBB_RMS_0001"))
      wr.writerows(export_data)
  myfile.close()

def main(_):

  # Problem parameters
  num_contexts = 2000

  # Data type in {linear, sparse_linear, mushroom, financial, jester,
  #                 statlog, adult, covertype, census, wheel}
  data_type = 'mushroom'

  # Create dataset
  sampled_vals = sample_data(data_type, num_contexts)
  dataset, opt_rewards, opt_actions, num_actions, context_dim = sampled_vals

  # Define hyperparameters and algorithms
  hparams = tf.contrib.training.HParams(num_actions=num_actions)


  hparams_greedy = tf.contrib.training.HParams(num_actions=num_actions,
                                            context_dim=context_dim,
                                            init_scale=0.3,
                                            activation=tf.nn.relu,
                                            layer_sizes=[100],
                                            batch_size=64,
                                            activate_decay=False,
                                            initial_lr=0.01,
                                            max_grad_norm=5.0,
                                            show_training=False,
                                            freq_summary=10,
                                            buffer_s=4096,
                                            initial_pulls=0,
                                            optimizer='RMS',
                                            reset_lr=True,
                                            lr_decay_rate=0.0,
                                            training_freq=1,
                                            training_epochs=64,
                                            p=1,
                                            q=3)

  hparams_greedy_1 = tf.contrib.training.HParams(num_actions=num_actions,
                                            context_dim=context_dim,
                                            init_scale=0.3,
                                            activation=tf.nn.relu,
                                            layer_sizes=[100],
                                            batch_size=64,
                                            activate_decay=False,
                                            initial_lr=0.01,
                                            max_grad_norm=5.0,
                                            show_training=False,
                                            freq_summary=10,
                                            buffer_s=4096,
                                            initial_pulls=0,
                                            optimizer='RMS',
                                            reset_lr=True,
                                            lr_decay_rate=0.0,
                                            training_freq=1,
                                            training_epochs=64,
                                            p=0.99,
                                            q=3)

  hparams_greedy_5 = tf.contrib.training.HParams(num_actions=num_actions,
                                            context_dim=context_dim,
                                            init_scale=0.3,
                                            activation=tf.nn.relu,
                                            layer_sizes=[100],
                                            batch_size=64,
                                            activate_decay=False,
                                            initial_lr=0.01,
                                            max_grad_norm=5.0,
                                            show_training=False,
                                            freq_summary=10,
                                            buffer_s=4096,
                                            initial_pulls=0,
                                            optimizer='RMS',
                                            reset_lr=True,
                                            lr_decay_rate=0.0,
                                            training_freq=1,
                                            training_epochs=64,
                                            p=0.95,
                                            q=3)

  hparams_bbb_Adam_001 = tf.contrib.training.HParams(num_actions=num_actions,
                                            context_dim=context_dim,
                                            init_scale=0.3,
                                            activation=tf.nn.relu,
                                            layer_sizes=[100],
                                            batch_size=64,
                                            activate_decay=False,
                                            initial_lr=0.01,
                                            max_grad_norm=5.0,
                                            show_training=False,
                                            freq_summary=10,
                                            buffer_s=4096,
                                            initial_pulls=0,
                                            optimizer='Adam',
                                            use_sigma_exp_transform=True,
                                            cleared_times_trained=0,
                                            initial_training_steps=64,
                                            noise_sigma=0.1,
                                            reset_lr=False,
                                            training_freq=1,
                                            training_epochs=64)

  hparams_bbb_Adam_0001 = tf.contrib.training.HParams(num_actions=num_actions,
                                            context_dim=context_dim,
                                            init_scale=0.3,
                                            activation=tf.nn.relu,
                                            layer_sizes=[100],
                                            batch_size=64,
                                            activate_decay=False,
                                            initial_lr=0.001,
                                            max_grad_norm=5.0,
                                            show_training=False,
                                            freq_summary=10,
                                            buffer_s=4096,
                                            initial_pulls=0,
                                            optimizer='Adam',
                                            use_sigma_exp_transform=True,
                                            cleared_times_trained=0,
                                            initial_training_steps=64,
                                            noise_sigma=0.1,
                                            reset_lr=False,
                                            training_freq=1,
                                            training_epochs=64)

  hparams_bbb_RMS_001 = tf.contrib.training.HParams(num_actions=num_actions,
                                            context_dim=context_dim,
                                            init_scale=0.3,
                                            activation=tf.nn.relu,
                                            layer_sizes=[100],
                                            batch_size=64,
                                            activate_decay=False,
                                            initial_lr=0.01,
                                            max_grad_norm=5.0,
                                            show_training=False,
                                            freq_summary=10,
                                            buffer_s=4096,
                                            initial_pulls=0,
                                            optimizer='RMS',
                                            use_sigma_exp_transform=True,
                                            cleared_times_trained=0,
                                            initial_training_steps=64,
                                            noise_sigma=0.1,
                                            reset_lr=False,
                                            training_freq=1,
                                            training_epochs=64)

  hparams_bbb_RMS_0001 = tf.contrib.training.HParams(num_actions=num_actions,
                                            context_dim=context_dim,
                                            init_scale=0.3,
                                            activation=tf.nn.relu,
                                            layer_sizes=[100],
                                            batch_size=64,
                                            activate_decay=False,
                                            initial_lr=0.001,
                                            max_grad_norm=5.0,
                                            show_training=False,
                                            freq_summary=10,
                                            buffer_s=4096,
                                            initial_pulls=0,
                                            optimizer='RMS',
                                            use_sigma_exp_transform=True,
                                            cleared_times_trained=0,
                                            initial_training_steps=64,
                                            noise_sigma=0.1,
                                            reset_lr=False,
                                            training_freq=1,
                                            training_epochs=64)


  algos = [
      PosteriorBNNSampling('Greedy', hparams_greedy, 'RMSProp'),
      PosteriorBNNSampling('Greedy_1', hparams_greedy_1, 'RMSProp'),
      PosteriorBNNSampling('Greedy_5', hparams_greedy_5, 'RMSProp'),

      PosteriorBNNSampling('BBB_Adam_0.01', hparams_bbb_Adam_001, 'Variational'),
      PosteriorBNNSampling('BBB_Adam_0.001', hparams_bbb_Adam_0001, 'Variational'),
      PosteriorBNNSampling('BBB_RMS_0.01', hparams_bbb_RMS_001, 'Variational'),
      PosteriorBNNSampling('BBB_RMS_0.001', hparams_bbb_RMS_0001, 'Variational'),

  ]

  # Run contextual bandit problem
  t_init = time.time()
  results = run_contextual_bandit(context_dim, num_actions, dataset, algos)
  _, h_rewards = results

  # Display results
  display_results(algos, opt_rewards, opt_actions, h_rewards, t_init, data_type)

if __name__ == '__main__':
  app.run(main)
