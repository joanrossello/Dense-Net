# network module
# adapted from: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F

# In this network we use the Leaky ReLU activation function with negative slope = 0.1

class NetLeaky(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5) # (input depth, output depth, kernel size)
        self.pool = nn.MaxPool2d(2, 2) # (kernel size, stride)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) 
        # after the previous convolutions we reduced the images to 5x5x16 --> depends on original size and conv you apply
        # these are the inputs to the linear layer, and we choose output to be 120
        self.fc2 = nn.Linear(120, 84) # we choose second output to be 84
        self.fc3 = nn.Linear(84, 10) # output needs to be 10 bc we have 10 different classes

    def forward(self, x):
        x = self.pool(F.leaky_relu(self.conv1(x), 0.1))
        x = self.pool(F.leaky_relu(self.conv2(x), 0.1))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.leaky_relu(self.fc1(x), 0.1)
        x = F.leaky_relu(self.fc2(x), 0.1)
        x = self.fc3(x)
        return x