import csv
from matplotlib import colors
import pandas as pd
import sys
sys.path.append('../')
from BayesBackpropagation import *
import math
import torch.optim as optim
import json

def generatePokemonData():

    data = []
    with open('pokemon_alopez247.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                r,g,b = colors.to_rgb(row[13])
                if row[3]!="":
                    data.append([row[1],row[13],r,g,b,row[3]])
                    data.append([row[1],row[13],r,g,b,row[2]])
                else:
                    data.append([row[1],row[13],r,g,b,row[2]])
                    
                line_count += 1
        print(f'Processed {line_count} lines.')

    data = np.asanyarray(data)[:,1:6]
    data = pd.DataFrame(data)
    data.columns = ['Colour', 'R', 'G','B','Type']
    types = data['Type'].unique()
    pokemonType = {}
    pokemon = {}

    for i in range(types.size):
        pokemonType[types[i]] = i
        pokemon[i] = types[i]
    data = data.replace({'Type': pokemonType})
    pokemonColors = data['Colour'].unique()
    """
    print(data[1].value_counts()/722)
    Blue      0.185596
    Brown     0.152355
    Green     0.109418
    Red       0.103878
    Grey      0.095568
    Purple    0.090028
    Yellow    0.088643
    White     0.072022
    Pink      0.056787
    Black     0.044321
    """

    testcolors = ['White','Pink','Black']
    train = data.loc[~data['Colour'].isin(testcolors)]
    test = data.loc[data['Colour'].isin(testcolors)]
    train_x = train.loc[:,['R', 'G','B']]
    train_y = train.drop(['Colour', 'R', 'G','B'],axis=1)
    test_x = test.loc[:,['R', 'G','B']]
    test_y = test.drop(['Colour', 'R', 'G','B'],axis=1)

    return  train_x,train_y,test_x,test_y, pokemon,pokemonColors

def train(net, optimizer, data, target, NUM_BATCHES):
    for i in range(NUM_BATCHES):
        net.zero_grad()
        x = data
        y = target
        loss = net.BBB_loss(x, y)
        loss.backward()
        optimizer.step()

print('Generating Data set.',)
train_x,train_y,test_x,test_y, pokemonType, pokemonColors = generatePokemonData()


#Hyperparameter setting
TRAIN_EPOCHS = 500
SAMPLES = 5
TEST_SAMPLES = 10
BATCH_SIZE = train_x.shape[0]
NUM_BATCHES = 1
TEST_BATCH_SIZE = test_x.shape[0]
CLASSES = 18
PI = 0.5
SIGMA_1 = torch.FloatTensor([math.exp(-0)])
SIGMA_2 = torch.FloatTensor([math.exp(-6)])
if torch.cuda.is_available():
    SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
    SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])

if torch.cuda.is_available():
    Var = lambda x, dtype=torch.cuda.FloatTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor
else:
    Var = lambda x, dtype=torch.FloatTensor: Variable(torch.from_numpy(x).type(dtype)) #converting data to tensor

X = torch.tensor(train_x.values.astype(np.float32))
Y = torch.tensor(train_y.values.astype(np.int)).view(-1)
X_test = torch.tensor(test_x.values.astype(np.float32))

#Training
print('Training Begins!')


#Declare Network
net = BayesianNetwork(inputSize = X.shape[1],\
                      CLASSES = CLASSES, \
                      layers=np.array([200,200]), \
                      activations = np.array(['relu','relu','softmax']), \
                      SAMPLES = SAMPLES, \
                      BATCH_SIZE = BATCH_SIZE,\
                      NUM_BATCHES = NUM_BATCHES,\
                      hasScalarMixturePrior = True,\
                      PI = PI,\
                      SIGMA_1 = SIGMA_1,\
                      SIGMA_2 = SIGMA_2).to(DEVICE)

#Declare the optimizer
#optimizer = optim.SGD(net.parameters(),lr=1e-4,momentum=0.9) #
optimizer = optim.Adam(net.parameters())

for epoch in range(TRAIN_EPOCHS):
    train(net, optimizer,data=X,target=Y,NUM_BATCHES=NUM_BATCHES)

print('Training Ends!')

results = {}
for color in pokemonColors:
    r,g,b = colors.to_rgb(color)
    temp = torch.tensor(np.asarray([r,g,b]).astype(np.float32))
    outputs = np.zeros(100)
    for i in range(100):
        output = net.forward(temp)
        output = output.max(1, keepdim=True)[1].data.numpy()
        outputs[i] = output[0][0]
    outputs = pd.DataFrame(outputs)
    outputs = outputs[0].value_counts()
    outputs = pd.DataFrame(outputs)
    outputs['Type'] = outputs.index.values.astype(np.int)
    outputs['Count'] = outputs[0]
    outputs = outputs.drop([0],axis=1)
    outputs = outputs.replace({'Type': pokemonType})
    results[color] = outputs.to_json(orient='values')

with open('PokemonResults.json', 'w') as fp:
    json.dump(results, fp, indent=4, sort_keys=True)
