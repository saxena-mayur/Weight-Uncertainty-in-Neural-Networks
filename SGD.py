# Credits: https://nextjournal.com/gkoehler/pytorch-mnist
# Which layers to drop?

import csv
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from LoadMNIST import load_mnist

if len(sys.argv) < 4:
    print('Call: python SGD.py [RATE 1-3] [UNITS 1-3] [EPOCHS]')
    sys.exit()

LEARNING_RATES = [1e-3, 1e-4, 1e-5]
LEARNING_RATE = LEARNING_RATES[int(sys.argv[1]) - 1]
HIDDEN_UNITS = 400 * int(sys.argv[2])  # tested in paper: 400, 800, 1200
ENABLE_DROPOUT = False
EPOCHS = int(sys.argv[3])
BATCH_SIZE = 128
USE_CUDA = torch.cuda.is_available()

torch.manual_seed(1)

print("----- FCN on MNIST -----")
print("Learning Rate: ", LEARNING_RATE)
print("Hidden Units : ", HIDDEN_UNITS)
print("Dropout      : ", ENABLE_DROPOUT)
print("CUDA         : ", USE_CUDA, '\n')
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Load MNIST
train_loader, valid_loader, test_loader = load_mnist(BATCH_SIZE)


class DropoutNetwork(nn.Module):
    def __init__(self):
        super(DropoutNetwork, self).__init__()
        self.fc0 = nn.Linear(28 * 28, HIDDEN_UNITS)
        self.fc1 = nn.Linear(HIDDEN_UNITS, HIDDEN_UNITS)
        self.fc2 = nn.Linear(HIDDEN_UNITS, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        if ENABLE_DROPOUT:
            x = F.dropout(x, p=0.2, training=self.training)
        h1 = F.relu(self.fc0(x))
        if ENABLE_DROPOUT:
            h1 = F.dropout(h1, p=0.5, training=self.training)
        h2 = F.relu(self.fc1(h1))
        if ENABLE_DROPOUT:
            h2 = F.dropout(h2, p=0.5, training=self.training)
        h3 = self.fc2(h2)

        return h3


model = DropoutNetwork()
# if torch.cuda.device_count() > 1:
#  print("Let's use", torch.cuda.device_count(), "GPUs!")
# dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
# model = nn.DataParallel( model )
model.to(DEVICE)

optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
loss_function = F.cross_entropy

# Keep track of progress
train_losses = []
valid_losses = []
test_losses = []
valid_correct = []
test_correct = []


def train(epoch):
    model.train()
    for batch_idx, (inputs, label) in enumerate(train_loader):
        inputs, label = inputs.to(DEVICE), label.to(DEVICE)

        # Forward pass
        optimizer.zero_grad()
        output = model(inputs)

        # Loss function
        loss = loss_function(output, label)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # if batch_idx % 100 == 0:
        #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(inputs), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader), loss.item()))

    train_losses.append(loss.item())


def test(epoch, data_loader, validation):
    model.eval()

    loss = 0.  # avg. loss over whole data set
    correct = 0.

    # Loop over WHOLE data set in batches
    with torch.no_grad():
        for inputs, label in data_loader:
            inputs, label = inputs.to(DEVICE), label.to(DEVICE)

            label = label.squeeze()
            output = model(inputs)

            loss += loss_function(output, label, reduction='sum').item()

            predict = output.data.max(1)[1]
            correct += predict.eq(label.data).cpu().sum()

    loss /= len(data_loader.dataset)

    if (validation):
        valid_losses.append(loss)
        valid_correct.append(correct)

        print('[Epoch {}] Loss: {:.4f}, Correct: {}'.format(
            epoch, loss, correct))
    else:
        test_losses.append(loss)
        test_correct.append(correct)


for epoch in range(EPOCHS):
    train(epoch + 1)
    test(epoch + 1, valid_loader, True)
    test(epoch + 1, test_loader, False)

suffix = str(LEARNING_RATE) + '_' + str(HIDDEN_UNITS)
torch.save(model.state_dict(), 'results/model_' + suffix + '.pth')


wr = csv.writer(open('results/results_' + suffix + '.csv', 'w'),
                delimiter=',', lineterminator='\n')
wr.writerow(['epoch', 'train_losses', 'valid_losses',
             'valid_correct', 'test_losses', 'test_correct'])
for i in range(EPOCHS):
    wr.writerow((i + 1, train_losses[i], valid_losses[i], int(
        valid_correct[i]), test_losses[i], int(test_correct[i])))
