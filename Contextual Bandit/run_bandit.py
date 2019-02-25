# Import Libraries

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import preprocessing
from sklearn import metrics
import random

import sys
sys.path.append('../')
from BayesBackpropagation import *

NB_STEPS = int(sys.argv[1])
print('running for {0} steps'.format(NB_STEPS))


# Import data from file
df = pd.read_csv(os.getcwd() + '/agaricus-lepiota.data', sep=',', header=None,
             error_bad_lines=False, warn_bad_lines=True, low_memory=False)

# Set pandas to output all of the columns in output
df.columns = ['class','cap-shape','cap-surface','cap-color','bruises','odor','gill-attachment',
         'gill-spacing','gill-size','gill-color','stalk-shape','stalk-root',
         'stalk-surf-above-ring','stalk-surf-below-ring','stalk-color-above-ring','stalk-color-below-ring',
         'veil-type','veil-color','ring-number','ring-type','spore-color','population','habitat']

# Split context from label
X = pd.DataFrame(df, columns=df.columns[1:len(df.columns)], index=df.index)
# Put the class values (0th column) into Y
Y = df['class']

# Transform labels into one-hot encoded array
le = preprocessing.LabelEncoder()
le.fit(Y)
y = le.transform(Y)

# Temporary variable to avoid error 
x_tmp = pd.DataFrame(X,columns=[X.columns[0]])

# Encode each feature column and add it to x_train 
for colname in X.columns:
    le.fit(X[colname])
    #print(colname, le.classes_)
    x_tmp[colname] = le.transform(X[colname])

# Produce mushroom array: 8124 mushrooms, each with 117 one-hot encoded features
oh = preprocessing.OneHotEncoder(categorical_features='all')
oh.fit(x_tmp)
x = oh.transform(x_tmp).toarray()

def get_reward(eaten, edible):
    # REWARDS FOR AGENT
    #  Eat poisonous mushroom
    if not eaten:
        return 0
    if eaten and edible:
        return 5
    elif eaten and not edible:
        return (5 if np.random.rand() > 0.5 else -35)

def oracle_reward(edible):
    return 5*edible    

def init_buffer():
    bufferX, bufferY = [], []
    for i in np.random.choice(range(len(x)), 4096):
        eat = np.random.rand()>0.5
        bufferX.append(np.concatenate((x[i], [1, 0] if eat else [0, 1])))
        bufferY.append(get_reward(eat, y[i]))
    return bufferX, bufferY

# Define some hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
print("Cuda available?: ",torch.cuda.is_available())

PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.FloatTensor([math.exp(-6)])

import torch.nn as nn
import torch.nn.functional as F
import sys

Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype))


class MushroomNet():
    def __init__(self, label='MushNet'):
        self.label = label
        self.epsilon = 0
        self.net = None
        self.bufferX, self.bufferY = init_buffer()
        self.loss, self.optimizer = None, None
        self.cum_regrets = [0]
    
    def expected_rewards(self, context, k=2):
        c_eat = Var(np.concatenate((context, [1, 0])))
        c_reject = Var(np.concatenate((context, [0, 1])))
        with torch.no_grad():
            r_eat = np.mean([self.net.forward(c_eat).numpy().reshape(1)[0] for _ in range(k)])
            r_reject = np.mean([self.net.forward(c_reject).numpy().reshape(1)[0] for _ in range(k)])
        return r_reject, r_eat

    def try_(self, mushroom):
        context, edible = x[mushroom], y[mushroom]
        r_reject, r_eat = self.expected_rewards(context, k=1)
        eaten = r_eat > r_reject
        if np.random.rand()<self.epsilon:
            eaten = (np.random.rand()<.5)
        reward = get_reward(eaten, edible)
        action = [1, 0] if eaten else [0, 1]
        self.bufferX.append(np.concatenate((context, action)))
        self.bufferY.append(reward)
        rg = oracle_reward(edible) - reward
        self.cum_regrets.append(self.cum_regrets[-1]+rg)
    
    def update(self, mushroom):
        self.try_(mushroom)
        bX = Var(np.array(self.bufferX[-4096:]))
        bY = Var(np.array(self.bufferY[-4096:]))
        for idx in np.split(np.random.permutation(range(4096)), 64):
            self.net.train()
            self.net.zero_grad()
            self.loss(bX[idx], bY[idx]).backward()
            self.optimizer.step()
            
class BBB_MNet(MushroomNet):
    def __init__(self, label):
        super().__init__(label)
        self.net = BayesianNetwork(inputSize = x.shape[1]+2,
                          CLASSES = 1, 
                          layers=np.array([100,100]), 
                          activations = np.array(['relu','relu','none']), 
                          SAMPLES = 1, 
                          BATCH_SIZE = 64,
                          NUM_BATCHES = 64).to(DEVICE)
        self.optimizer = optim.Adam(self.net.parameters())
        self.loss = lambda data, target:self.net.sample_elbo(data, target)[0]
        
        

class EpsGreedyMlp(MushroomNet):
    def __init__(self, label, epsilon=0):
        super().__init__(label)
        self.epsilon = epsilon
        self.net = nn.Sequential(
        nn.Linear(x.shape[1]+2, 100), nn.ReLU(),
        nn.Linear(100, 100), nn.ReLU(),
        nn.Linear(100, 1))
        self.bufferX, self.bufferY = init_buffer()
        self.optimizer = optim.Adam(self.net.parameters())
        self.mse = nn.MSELoss()
        self.loss = lambda data, target: self.mse(self.net.forward(data), target)

mushroom_nets = {'bbb':BBB_MNet(label = 'Bayes By BackProp'),
                 'e0':EpsGreedyMlp(epsilon=0, label = 'Greedy'),
                 'e1':EpsGreedyMlp(epsilon=0.01, label = '1% Greedy'),
                 'e5':EpsGreedyMlp(epsilon=0.05, label = '5 %Greedy')}


for _ in tqdm(range(NB_STEPS)):
    mushroom = np.random.randint(len(x))
    for j, (key, net) in enumerate(mushroom_nets.items()):
        net.update(mushroom)

import pandas as pd
df = pd.DataFrame.from_dict({net.label: net.cum_regrets for i, net in mushroom_nets.items()})
df.to_csv('mushroom_regrets.csv')