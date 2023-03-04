cd import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


KERNAL = 5

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    # Layer 1 sees a 1x50x50 image tensors
    self.conv1 = nn.Conv2d(1, 32, KERNAL) 
    # Layer 2 sees a 32x46x46
    self.conv2 = nn.Conv2d(32, 64, KERNAL)
    # Layer 3 sees a 64x42x42
    self.conv3 = nn.Conv2d(64, 128, KERNAL)
    #output of third conv layer is 128x38x38
    # Max pooling
    self.pool1 = nn.MaxPool2d((2,2))
    self.pool2 = nn.MaxPool2d((2,2))
    #self.pool3 = nn.MaxPool2d((2,2))
    self.fc1 = nn.Linear(128*5*5, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 1)
    self.dropout = nn.Dropout(0.25)

  def forward(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = F.relu(self.conv3(x))
    x = torch.flatten(x, start_dim=1)
    x = self.dropout(x)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = (self.fc3(x))
    x = F.sigmoid(x)
    return x