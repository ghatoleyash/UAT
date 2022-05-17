import torch
import torch.nn as nn
import torch.nn.functional as F

class WideNet(nn.Module):
    def __init__(self, hiddenNeurons):
        super(WideNet, self).__init__()
        self.linear1 = nn.Linear(1, hiddenNeurons)
        self.linear2 = nn.Linear(hiddenNeurons, hiddenNeurons)
        self.output = nn.Linear(hiddenNeurons, 1)
    
    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.output(x))
        return x