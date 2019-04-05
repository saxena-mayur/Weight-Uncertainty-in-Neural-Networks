import csv
from matplotlib import colors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import sys
sys.path.append('../')
from BayesBackpropagation import *
import math
import torch.optim as optim
import json
import colorsys

def loadPokemonColours():
    with open('median_values.json') as f:
        data = json.load(f)
    return data

def generatePokemonData(NUM_BATCHES):

    colourMap = loadPokemonColours()
    data = []
    with open('data/pokemon.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                #noise = np.random.normal(0, 0.02, size=3)
                col = colourMap[row[0]]
                r = col['R']/255.
                g = col['G']/255.
                b = col['B']/255.
                #r,g,b = colorsys.rgb_to_hsv(r,g,b)
                #r,g,b = colors.to_rgb(row[13]) + noise
                if row[3]!="":
                    data.append([row[1],row[13],r,g,b,row[3]])
                    data.append([row[1],row[13],r,g,b,row[2]])
                else:
                    data.append([row[1],row[13],r,g,b,row[2]])
                
            line_count += 1
    data = np.asanyarray(data)[:,1:6]
    data = pd.DataFrame(data)
    data.columns = ['Colour', 'R', 'G','B','Type']
    types = data['Type'].unique()
    pokemonType = {}
    pokemon = {}
    """
    temp = data.groupby(['Colour', 'Type']).size()
    temp = temp.reset_index(level=['Colour','Type'])
    temp = temp.pivot(index='Colour', columns='Type')
    temp.to_csv(r'./DataDistribution.csv')
    """

    for i in range(types.size):
        pokemonType[types[i]] = i
        pokemon[i] = types[i]
    data = data.replace({'Type': pokemonType})
    
    pokemonColors = data['Colour'].unique()
    train_x = data.loc[:,['R', 'G','B']]
    train_y = data.drop(['Colour', 'R', 'G','B'],axis=1)
    
    X = np.array_split(train_x,NUM_BATCHES)
    Y = np.array_split(train_y,NUM_BATCHES)
    
    return  X,Y,pokemon,pokemonColors

def train(net, optimizer, data, target, NUM_BATCHES):
    for i in range(NUM_BATCHES):
        net.zero_grad()
        x = torch.tensor(data[i].values.astype(np.float32)).to(DEVICE)
        y = torch.tensor(target[i].values.astype(np.int)).view(-1).to(DEVICE)
        loss = net.BBB_loss(x, y)
        loss.backward()
        optimizer.step()

def trainBBB(train_x,train_y,TRAIN_EPOCHS,NUM_BATCHES):
    #Hyperparameter setting
    SAMPLES = 20
    BATCH_SIZE = train_x[0].shape[0]
    CLASSES = 18
    INPUT_SIZE = 3
    PI = 0.5
    SIGMA_1 = torch.FloatTensor([math.exp(-0)])
    SIGMA_2 = torch.FloatTensor([math.exp(-6)])
    if torch.cuda.is_available():
        SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
        SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])
 
    #Training
    print('Training Begins!')


    #Declare Network
    net = BayesianNetwork(inputSize = INPUT_SIZE,\
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
    optimizer = optim.SGD(net.parameters(),lr=1e-4,momentum=0.9) #
    #optimizer = optim.Adam(net.parameters())

    for epoch in range(TRAIN_EPOCHS):
        train(net, optimizer,data=train_x,target=train_y,NUM_BATCHES=NUM_BATCHES)
        print('Epoch: ',epoch)

    print('Training Ends!')

    return net

def test(net, colors_pokemon, pokemonType, TEST_SAMPLES):
    results = {}
    for color in colors_pokemon:
        r,g,b = colors.to_rgb(color)
        temp = torch.tensor(np.asarray([r,g,b]).astype(np.float32)).to(DEVICE)
        result = []
        for i in range(TEST_SAMPLES):
            output = net.forward(temp)
            a = output[0].data.cpu().numpy()
            result.append(np.exp(a) / (np.exp(a)).sum())
        result = np.mean(result, axis = 0)
        result = pd.DataFrame(result)
        result['Type'] = result.index.values.astype(np.int)
        result['Probability'] = result[0]
        result = result.drop([0],axis=1)
        result = result.replace({'Type': pokemonType})
        result = result.sort_values(by='Probability', ascending=False)
        results[color] = result.to_json(orient='values')
    
    return results


TRAIN_EPOCHS = 500
TEST_SAMPLES = 10
NUM_BATCHES = 10
newColors = ['Orange','Lime','Maroon','Silver','Navy','Magenta','Aqua','Gold','Chocolate','Olive']

print('Generating Data set.')
train_x,train_y,pokemonType, pokemonColors = generatePokemonData(NUM_BATCHES)

with open('data/PokemonTypeMap.json', 'w') as fp:
        json.dump(pokemonType, fp, indent=4, sort_keys=True)

net = trainBBB(train_x,train_y,TRAIN_EPOCHS,NUM_BATCHES)

"""
print('Testing begins!')
results = {}
results['original'] = test(net, pokemonColors, pokemonType, TEST_SAMPLES)
results['newData'] = test(net, newColors, pokemonType, TEST_SAMPLES)

with open('PokemonResults.json', 'w') as fp:
        json.dump(results, fp, indent=4, sort_keys=True)
print('Testing ends! Results saved.')
"""

#Save the trained model
torch.save(net.state_dict(), './Model.pth')
