import bayes
from utils import Flatten
import torch.nn as nn

class MLP(nn.Sequential):
    def __init__(self, inputs, outputs, hidden=100):
        super().__init__(
            Flatten(inputs),
            nn.Linear(inputs, hidden),
            nn.Softplus(),
            #nn.Linear(hidden, hidden),
            #nn.Softplus(),
            nn.Linear(hidden, outputs))
    
class LeNet(nn.Sequential):
    def __init__(self, dropout=0):
        super().__init__(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(16*5*5),
            nn.Linear(16*5*5, 120), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(120, 84), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(84, 10))
    

class SmallAlexNet(nn.Sequential):
    def __init__(self, inputs=3, outputs = 10):
        super().__init__(
            nn.Conv2d(inputs, 64, kernel_size=11, stride=4, padding=5),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.Softplus(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.Softplus(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(1 * 1 * 128), 
            nn.Linear(128, outputs))

# Bigger Model
class AlexNet(nn.Sequential):
    def __init__(self, num_classes=10, dropout = 0):
        super().__init__(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            Flatten(256*2*2),
            nn.Dropout(dropout),
            nn.Linear(256 * 2 * 2, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, 512),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes))