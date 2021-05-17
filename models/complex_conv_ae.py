import torch
from torch import nn


class ComplexConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ComplexConvAutoEncoder, self).__init__()
        self.encode1 = nn.Conv1d(1, 4, 501, padding=250)
        self.encode2 = nn.Conv1d(4, 8, 251, padding=125)
        self.encode3 = nn.Conv1d(8, 16, 125, padding=62)
        self.decode1 = nn.Conv1d(16, 8, 125, padding=62)
        self.decode2 = nn.Conv1d(8, 4, 251, padding=125)
        self.decode3 = nn.Conv1d(4, 1, 501, padding=250)
        
        self.pool = nn.MaxPool1d(10, stride=10, return_indices=True)
        self.unpool = nn.MaxUnpool1d(10, stride=10)
        self.activation_func = nn.ReLU()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.activation_func(self.encode1(x))
        y1 = x.size()
        x, p1 = self.pool(x)
        x = self.activation_func(self.encode2(x))
        y2 = x.size()
        x, p2 = self.pool(x)
        x = self.activation_func(self.encode3(x))
        x = self.activation_func(self.decode1(x))
        x = self.unpool(x, p2, output_size=y2)
        x = self.activation_func(self.decode2(x))
        x = self.unpool(x, p1, output_size=y1)
        x = self.decode3(x)
        return torch.squeeze(x)