from matplotlib import colors
import pandas as pd
import colorsys
import csv
import numpy as np
import copy
import matplotlib.pyplot as plt
import json
plt.switch_backend('agg')

def loadPokemonColours():
    with open('median_values.json') as f:
        data = json.load(f)
    return data

colourMap = loadPokemonColours()
data = []
with open('pokemon.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count != 0:
            col = colourMap[row[0]]
            r = col['R']/255.
            g = col['G']/255.
            b = col['B']/255.
            h,s,v = colorsys.rgb_to_hsv(r,g,b)
            data.append([h,s,v,r,g,b]) 
        line_count += 1

data = np.asanyarray(data)
data = pd.DataFrame(data)
data = data[data[1] > 0.5]
data = data.sort_values(by=[0],ascending = False) #To form the spectrum
print(data)

data = data.drop([0,1,2],axis=1)
data = data.drop_duplicates()
data = data.as_matrix()
weight = np.ones(data.shape[0])

plt.figure(figsize=(9,8))

patches, texts = plt.pie(weight, colors=data, startangle=90)
plt.axis('equal')
plt.savefig('./Results/TrainingColours.png',transparent=True)
