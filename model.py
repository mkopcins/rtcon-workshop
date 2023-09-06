import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=2) # (16,14,14)
         )
        self.conv2 = nn.Sequential( # (16,14,14)
             nn.Conv2d(16, 32, 5, 1, 2), # (32,14,14)
             nn.ReLU(),
             nn.MaxPool2d(2) # (32,7,7)
         )
        self.out = nn.Linear(32*7*7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1) # (batch, 32,7,7) -> (batch, 32*7*7)
        output = self.out(x)
        return output
