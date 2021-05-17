import torch
from torch import nn


class SimpleConvAutoEncoder(nn.Module):
    def __init__(self):
        super(SimpleConvAutoEncoder, self).__init__()
        self.encode1 = nn.Conv1d(1, 4, 501, padding=250)
        self.decode1 = nn.Conv1d(4, 1, 501, padding=250)
        
        self.pool = nn.MaxPool1d(10, stride=10, return_indices=True)
        self.unpool = nn.MaxUnpool1d(10, stride=10)
        self.activation_func = nn.ReLU()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.activation_func(self.encode1(x))
        y1 = x.size()
        x, p1 = self.pool(x)
        x = self.unpool(x, p1, output_size=y1)
        x = self.decode1(x)
        return torch.squeeze(x)