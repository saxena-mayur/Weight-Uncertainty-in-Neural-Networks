import math
import sys
sys.path.append('../')
from BayesBackpropagation import *
import numpy as np
import copy
import matplotlib.pyplot as plt
import json
from matplotlib import colors
import pandas as pd
import colorsys
import csv
plt.switch_backend('agg')

def loadPokemonModel(filename):
    # Hyperparameter declaration
    BATCH_SIZE = 20
    TEST_BATCH_SIZE = 1000
    CLASSES = 18
    TRAIN_EPOCHS = 1000
    SAMPLES = 20
    PI = 0.5
    SIGMA_1 = torch.FloatTensor([math.exp(-0)])
    SIGMA_2 = torch.FloatTensor([math.exp(-6)])
    if torch.cuda.is_available():
        SIGMA_1 = torch.cuda.FloatTensor([math.exp(-0)])
        SIGMA_2 = torch.cuda.FloatTensor([math.exp(-6)])
    INPUT_SIZE = 3

    LAYERS = np.array([200,200])
    NUM_BATCHES = 0
    ACTIVATION_FUNCTIONS = np.array(['relu','relu','softmax'])
    net = BayesianNetwork(inputSize = INPUT_SIZE,\
                            CLASSES = CLASSES, \
                            layers=LAYERS, \
                            activations = ACTIVATION_FUNCTIONS, \
                            SAMPLES = SAMPLES, \
                            BATCH_SIZE = BATCH_SIZE,\
                            NUM_BATCHES = NUM_BATCHES,\
                            hasScalarMixturePrior = True,\
                            PI = PI,\
                            SIGMA_1 = SIGMA_1,\
                            SIGMA_2 = SIGMA_2).to(DEVICE)

    net.load_state_dict(torch.load(filename, map_location='cpu'))
    net.eval()
    return net

def loadPokemonColours():
    with open('median_values.json') as f:
        data = json.load(f)
    
    cols = []
    for key in data:
        temp  = data[key]
        r = [temp['R']/255,temp['G']/255,temp['B']/255]
        cols.append(r)
    cols = np.array(cols)
    return cols

def loadPokemonTypeMap():
    with open('PokemonTypeMap.json') as f:
        pokemonType = json.load(f)
    for x in range(18):
        pokemonType[x] = pokemonType.pop(str(x))
    return pokemonType

def test(net, r,g,b, pokemonType, TEST_SAMPLES):
    temp = torch.tensor(np.asarray([r,g,b]).astype(np.float32)).to(DEVICE)
    result = []
    for i in range(TEST_SAMPLES):
        output = net.forward(temp)
        a = output[0].data.cpu().numpy()
        result.append(np.exp(a) / (np.exp(a)).sum())
        
    mean = np.mean(result, axis = 0)
    std = np.mean(np.std(result, axis = 0))
    result = pd.DataFrame({'Mean.Probability': mean,'Std':std})
    result['Type'] = result.index.values.astype(np.int)
    result = result.replace({'Type': pokemonType})
    result = result.sort_values(by='Mean.Probability', ascending=False)
    #print(result)
    temp = result.iloc[1]
    result = result.iloc[0]
    result['TopTwo'] = result['Type']+"/"+temp['Type']
    result['Type'] = result['Type']+"("+ str("{0:.2f}".format(result['Mean.Probability']))+")"
    return result
    
def closest_node_distance(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sqrt(np.sum((nodes - node)**2, axis=1))
    return np.min(dist_2)

def generateGraph(H,s,v,colours):
    distance = []
    col = [] #To store Pokemon's colour in RGB
    variance = [] #To store standard deviation for a colour
    TEST_SAMPLES= 2

    for h in H:
        r,g,b= colorsys.hsv_to_rgb(h, s, v) #convert hsv to rgb
        col.append((r,g,b))
        temp = test(net,r,g,b,pokemonType,TEST_SAMPLES)
        distance.append(closest_node_distance([r,g,b],colours))
        variance.append("{0:.4f}".format(temp['Std']))
    
    r = pd.DataFrame({'Distance':distance,'Colour':col,'Hue':H,'Std':variance})
    r = r.sort_values(by=['Hue'], ascending=False) #To form the spectrum
    r = r.drop(['Hue'],axis=1)
    r = r.drop_duplicates()
    return r

net = loadPokemonModel('./Model.pth')
pokemonType = loadPokemonTypeMap()

data  = loadPokemonColours()

#Initialize the HSV values
H = np.arange(0, 1.01, 0.01)
S = np.arange(0, 1.01, 0.25)
V = np.arange(0, 1.01, 0.25)

r = pd.DataFrame()

for s in S:
    for v in V:
        if r.empty:
            r = generateGraph(H,s,v,data)
        else:
            r = r.append(generateGraph(H,s,v,data))

r = r.sort_values(by=['Distance'], ascending=False) 


print('Displaying graph')

y = pd.to_numeric(r.Std).values
x = pd.to_numeric(r.Distance).values
c = r.Colour.values 

plt.xlabel('Distance from the nearest Training data point')
plt.ylabel('Standard deviation in prediction')
plt.plot(x,y)
plt.savefig('./Results/Uncertainty.png')
plt.clf()