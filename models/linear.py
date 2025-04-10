import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, feat_dim=256, num_classes=9):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out


class LinearRegress(nn.Module):
    def __init__(self, feat_dim=512):
        super(LinearRegress, self).__init__()
        self.hidden = nn.Linear(feat_dim, 256)
        # self.act = nn.ReLU()
        self.linear = nn.Linear(256, 1)

    def forward(self, x):
        out = self.hidden(x)
        # out = self.act(out)
        out = self.linear(out).view(-1, 1)
        return out

