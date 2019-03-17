import os
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from bayes import BayesWrapper
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from utils import DEVICE, ScaleMixtureGaussian, Flatten

torch.manual_seed(0)

valid_size = 1 / 6
batch_size = 125

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255. / 126.),
])

train_data = torchvision.datasets.MNIST(
    root='../data',
    train=True,
    download=True,
    transform=transform)

test_data = torchvision.datasets.MNIST(
    root='../data',
    train=False,
    download=True,
    transform=transform)

# obtain training indices that will be used for validation
num_train = len(train_data)
indices = list(range(num_train))
split = int(valid_size * num_train)
train_idx, valid_idx = indices[split:], indices[:split]

# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    sampler=train_sampler,
    num_workers=1)

valid_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch_size,
    sampler=valid_sampler,
    num_workers=1)

test_loader = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch_size,
    num_workers=1)

classes = range(10)
TEST_SAMPLES = 2

def accuracy(outputs, labels):
    _, preds = outputs.max(1)
    correct = torch.eq(preds, labels).sum().item()
    return correct / len(labels)


def test(model):
    test_accs = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            test_accs.append(accuracy(outputs, labels))
    print('\nTest Error: {:.2f} %'.format(100 - 100 * np.mean(test_accs)))


def bayes_train(name, model, n_epochs=100, n_samples=2):
    model.train()
    m = len(train_loader)

    it_accs = []
    acc = 0.

    for epoch in range(n_epochs):
        history = dict(val_acc=[])

        # Train
        for batch_id, (images, labels) in enumerate(train_loader):
            beta = 2 ** (m - (batch_id + 1)) / (2 ** m - 1)
            # beta = 1 / (m)

            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = [model(images) for _ in range(n_samples)]
            kl, xe, loss = model.step(outputs, labels, beta)

        # Validation
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = torch.zeros(TEST_SAMPLES, batch_size, len(classes)).to(DEVICE)
                for i in range(TEST_SAMPLES):
                    outputs[i] = model(images)
                output = outputs.mean(0)

                #output = model(images)
                history['val_acc'].append(1 - accuracy(output, labels))

        # Save values
        # for key, values in history.items():
        #    print('MNIST/{0}/{1}'.format(name, key),
        #          np.mean(values), epoch + 1)
        acc = np.mean(history['val_acc'])
        print(acc)
        it_accs.append(acc)
    return model, np.mean(it_accs), acc


N_EPOCHS = 1
SAMPLES = 5
HIDDEN = 400

mlp = nn.Sequential(Flatten(28 * 28),
                    nn.Linear(28 * 28, HIDDEN),
                    nn.ReLU(),
                    nn.Linear(HIDDEN, HIDDEN),
                    nn.ReLU(),
                    nn.Linear(HIDDEN, 10))


# 400  1e-3 .98 .25 1. 7.
# 800  1e-3 .99 .75 1. 7.
# 1200 1e-3 .99 .75 0. 7.

params = {
    'lr': [1e-3],
    'momentum': [.98],
    'pi': [0.25], #, 0.5, 0.75],
    's1': [1.], #, 0., 2.],
    's2': [7.], #6., 8.],
} # keys must be sorted
keys = sorted(params)
paramCombinations = list(product(*(params[key] for key in keys)))

results = []
i = 0

for lr, momentum, pi, s1, s2 in tqdm(paramCombinations):
    i+=1
    prior_nll = ScaleMixtureGaussian(PI=pi,
                                     SIGMA_1=np.exp(-s1),
                                     SIGMA_2=np.exp(-s2))

    bayesnet = BayesWrapper(name='BayesMLP',
                            net=mlp.to(DEVICE),
                            prior_nll=prior_nll,
                            lr=lr,
                            momentum=momentum)

    model, mean_acc, final_acc = bayes_train('Bayes MLP', bayesnet, n_epochs=N_EPOCHS, n_samples=SAMPLES)
    result = (i, lr, pi, s1, s2, momentum, mean_acc, final_acc)
    print(result)
    results.append(result)

    path = 'BBB_hyperparams_' + str(HIDDEN)
    np.savetxt(path + '.csv', results)
    torch.save(model.bayes_params, path + '.pth')


exec(open("WeightPruningCNN.py").read())

for root, dirs, files in os.walk("./"):
    for file in files:
        if file.startswith('BBB_hyperparams_400') and file.endswith(".pth"): # +str(HIDDEN_UNITS)
            print(file)
            bayesnet.load_state_dict(torch.load("./" + file))
            test(bayesnet)