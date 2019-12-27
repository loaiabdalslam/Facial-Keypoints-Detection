## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # input image : (1, 224, 224) - graysclae squared image
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.bn1 = nn.BatchNorm2d(32)
        self.bool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Droupout(p=0.1)

        self.conv2 = nn.Conv2d(32, 64, 3)
        self.bn2 = nn.BatchNorm2d(64)
        self.bool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Droupout(p=0.2)

        self.conv3 = nn.Conv2d(64, 128, 2)
        self.bn3 = nn.BatchNorm2d(128)
        self.bool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Droupout(p=0.3)

        self.conv4 = nn.Conv2d(128, 256, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.bool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Droupout(p=0.4)

        self.fc1 = nn.Linear(43264, 1024)
        self.bn5 = nn.BatchNorm1d(1024)
        self.dorpout5 = nn.Droupout(p=0.5)

        self.fc2 = nn.Linear(1024, 1024)
        self.bn6 = nn.BatchNorm1d(1024)
        self.dorpout6 = nn.Droupout(p=0.5)

        self.fc3 = nn.Linear(1024, 136)

        I.xavier_normal(self.fc1.weights.data)
        I.xavier_normal(self.fc2.weights.data)
        I.xavier_normal(self.fc3.weights.data)


    def forward(self, x):

        return x
