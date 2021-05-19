import torch
from torch import nn


class ComplexEncoder(nn.Module):
    def __init__(self):
        super(ComplexEncoder, self).__init__()
        self.encode1 = nn.Linear(220500, 250)
        self.encode2 = nn.Linear(250, 100)
        self.encode3 = nn.Linear(100, 50)
        
        self.activation_func = nn.ReLU()

    def forward(self, stimulus):
        x = self.activation_func(self.encode1(stimulus))
        x = self.activation_func(self.encode2(x))
        x = self.activation_func(self.encode3(x))
        return x


class ComplexDecoder(nn.Module):
    def __init__(self):
        super(ComplexDecoder, self).__init__()
        self.decode1 = nn.Linear(50, 100)
        self.decode2 = nn.Linear(100, 250)
        self.decode3 = nn.Linear(250, 220500)
        
        self.activation_func = nn.ReLU()

    def forward(self, stimulus):
        x = self.activation_func(self.decode1(stimulus))
        x = self.activation_func(self.decode2(x))
        x = self.decode3(x)
        return x


class ComplexAutoEncoder(nn.Module):
    def __init__(self):
        super(ComplexAutoEncoder, self).__init__()
        self.encoder = ComplexEncoder()
        self.decoder = ComplexDecoder()

    def forward(self, stimulus):
        x = self.encoder(stimulus)
        x = self.decoder(x)
        return x