# Import Libraries

import math
import pandas as pd
import os
import torch.optim as optim
from sklearn import preprocessing
from tqdm import tqdm

import sys
sys.path.append('../../')
from BayesBackpropagation import *

#### CUDA NOT YET IMPLEMENTED - DISABLE IN BayesBackpropagation.py ###

NB_STEPS = 50000
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
        return 5 if np.random.rand() > 0.5 else -35

def oracle_reward(edible):
    return 5*edible

# Initialise data matrix, each element includes an action, context and reward

def init_data():
    contexts, types, optimal_rewards = [], [], []
    for i in np.random.choice(range(len(x)), 10000):
        contexts.append(x[i])
        types.append(y[i])
        optimal_rewards.append(oracle_reward(y[i]))
    return contexts, types, optimal_rewards

contexts, types, optimal_rewards = init_data()


# Define some hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
print("Cuda available?: ",torch.cuda.is_available())

PI = 0.75
SIGMA_1 = torch.FloatTensor([math.exp(-1)])
SIGMA_2 = torch.FloatTensor([math.exp(-6)])

import torch.nn as nn
import torch.nn.functional as F
import sys

Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype))


class MushroomNet():
    def __init__(self, label = 'MushNet', n_weight_sampling=2):
        self.label = label
        self.n_weight_sampling = n_weight_sampling
        self.epsilon = 0
        self.net = None
        self.loss, self.optimizer = None, None
        self.cum_regrets = [0]
        self.bufferX = []
        self.bufferY = []
    
    # Use NN to decide next action
    def try_ (self, mushroom):
        samples = self.n_weight_sampling
        context, edible = x[mushroom], y[mushroom]
        try_eat = Var(np.concatenate((context, [1, 0])))
        try_reject = Var(np.concatenate((context, [0, 1])))
        
        # Calculate rewards using model
        with torch.no_grad():
            r_eat = np.mean([self.net.forward(try_eat).numpy() for _ in range(samples)])
            r_reject = np.mean([self.net.forward(try_reject).numpy() for _ in range(samples)])
        
        # Take random action for epsilon greedy agents, calculate agent's reward
        eaten = r_eat > r_reject
        if np.random.rand()<self.epsilon:
            eaten = (np.random.rand()<.5)
        agent_reward = get_reward(eaten, edible)
        
        # Get rewards and update buffer
        if eaten:
            action = [1, 0]
        else:
            action = [0, 1]
        self.bufferX.append(np.concatenate((context, action)))
        self.bufferY.append(agent_reward)
        
        # Calculate regret
        oracle = oracle_reward(edible)
        regret = oracle - agent_reward
        self.cum_regrets.append(self.cum_regrets[-1]+regret)


    # Function for generating a minibatch
    def generate_minibatch(self, trainingX, trainingY):
        bX = []
        bY = []
        for i in range(64):
            random = np.random.randint(0, len(trainingX))
            bX.append(trainingX[random])
            bY.append(trainingY[random])
            return bX, bY
    
    # Feed next mushroom
    def update(self, mushroom):
        self.try_(mushroom)
        for minibatch in range(64):
            bX, bY = self.generate_minibatch(self.bufferX[-4096:], self.bufferY[-4096:])
            self.net.zero_grad()
            self.loss(Var(np.asarray(bX)), Var(np.asarray(bY))).backward()
            self.optimizer.step()


# Class for BBB agent
class BBB_MNet(MushroomNet):
    def __init__(self, label, lr=0, **kwargs):
        super().__init__(label, **kwargs)
        self.net = BayesianNetwork(inputSize = x.shape[1]+2,
                                   CLASSES = 1,
                                   layers=np.array([100,100]),
                                   activations = np.array(['relu','relu','none']),
                                   SAMPLES = 2,
                                   BATCH_SIZE = 64,
                                   NUM_BATCHES = 64,
                                   hasScalarMixturePrior=True,
                                   PI=PI,
                                   SIGMA_1 = SIGMA_1,
                                   SIGMA_2 = SIGMA_2,
                                   GOOGLE_INIT=True,
                                   ).to(DEVICE)
        self.lr = lr
        self.optimizer = optim.Adam(self.net.parameters(), lr = self.lr)
        self.loss = lambda data, target:self.net.BBB_loss(data, target)


# Class for Greedy agents
class EpsGreedyMlp(MushroomNet):
    def __init__(self, epsilon=0, lr=0, **kwargs):
        super().__init__(**kwargs)
        self.n_weight_sampling = 1
        self.epsilon = epsilon
        self.lr = lr
        self.net = nn.Sequential(
                                 nn.Linear(x.shape[1]+2, 100), nn.ReLU(),
                                 nn.Linear(100, 100), nn.ReLU(),
                                 nn.Linear(100, 1))
        self.optimizer = optim.SGD(self.net.parameters(), lr = self.lr)
        self.mse = nn.MSELoss(reduction = 'sum')
        self.loss = lambda data, target: self.mse(self.net.forward(data), target)


mushroom_nets = {'bbb0.001':BBB_MNet(label = 'BBB 0.001', lr = 0.001),
    'bbb0.0001':BBB_MNet(label = 'BBB 0.0001', lr = 0.0001),
    'bbb0.00001':BBB_MNet(label = 'BBB 0.00001', lr = 0.00001),
    
    'e0.001':EpsGreedyMlp(epsilon=0, label = 'Greedy 0.001', lr = 0.001),
    'e0.0001':EpsGreedyMlp(epsilon=0, label = 'Greedy 0.0001', lr = 0.0001),
    'e0.00001':EpsGreedyMlp(epsilon=0, label = 'Greedy 0.00001', lr = 0.00001),
        
    'e1.001':EpsGreedyMlp(epsilon=0.01, label = '1% Greedy 0.001', lr = 0.001),
    'e1.0001':EpsGreedyMlp(epsilon=0.01, label = '1% Greedy 0.0001', lr = 0.0001),
    'e1.00001':EpsGreedyMlp(epsilon=0.01, label = '1% Greedy 0.00001', lr = 0.00001),
            
    'e5.001':EpsGreedyMlp(epsilon=0.05, label = '5% Greedy 0.001', lr = 0.001),
    'e5.0001':EpsGreedyMlp(epsilon=0.05, label = '5% Greedy 0.0001', lr = 0.0001),
    'e5.00001':EpsGreedyMlp(epsilon=0.05, label = '5% Greedy 0.00001', lr = 0.00001),}


for _ in tqdm(range(NB_STEPS)):
    mushroom = np.random.randint(len(x))
    for j, (key, net) in enumerate(mushroom_nets.items()):
        net.update(mushroom)

import pandas as pd
df = pd.DataFrame.from_dict({net.label: net.cum_regrets for i, net in mushroom_nets.items()})
df.to_csv('mushroom_regrets_empty_init.csv')
