import torch
from torch import nn


class SimpleConvEncoder(nn.Module):
    def __init__(self):
        super(SimpleConvEncoder, self).__init__()
        self.encode1 = nn.Conv1d(1, 4, 501, padding=250)
        
        self.pool = nn.MaxPool1d(10, stride=10, return_indices=True)
        self.activation_func = nn.ReLU()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.activation_func(self.encode1(x))
        y1 = x.size()
        x, p1 = self.pool(x)
        return x, (y1, p1)
        

class SimpleConvDecoder(nn.Module):
    def __init__(self):
        super(SimpleConvDecoder, self).__init__()
        self.decode1 = nn.Conv1d(4, 1, 501, padding=250)
        
        self.unpool = nn.MaxUnpool1d(10, stride=10)
        self.activation_func = nn.ReLU()

    def forward(self, x, y1, p1):
        x = self.unpool(x, p1, output_size=y1)
        x = self.decode1(x)
        return torch.squeeze(x)
        

class SimpleConvAutoEncoder(nn.Module):
    def __init__(self):
        super(SimpleConvAutoEncoder, self).__init__()
        self.encoder = SimpleConvEncoder()
        self.decoder = SimpleConvDecoder()

    def forward(self, x):
        x, (y1, p1) = self.encoder(x)
        x = self.decoder(x, y1, p1)
        return x