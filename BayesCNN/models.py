import bayes
from utils import Flatten
import torch.nn as nn

class AlexNet(nn.Sequential):
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

class AlexNetDropout(nn.Sequential):
    def __init__(self, inputs=3, outputs = 10):
        super().__init__(
            nn.Conv2d(inputs, 64, kernel_size=11, stride=4, padding=5),
            nn.Softplus(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.Softplus(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.Softplus(),
            nn.Dropout(p=0.5),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.Softplus(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.Softplus(),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            Flatten(128), 
            nn.Linear(128, outputs))
