from torch import nn


class SimpleEncoder(nn.Module):
    def __init__(self):
        super(SimpleEncoder, self).__init__()
        self.encode1 = nn.Linear(220500, 50)
        
        self.activation_func = nn.ReLU()

    def forward(self, stimulus):
        x = self.activation_func(self.encode1(stimulus))
        return x


class SimpleDecoder(nn.Module):
    def __init__(self):
        super(SimpleDecoder, self).__init__()
        self.decode1 = nn.Linear(50, 220500)
        
        self.activation_func = nn.ReLU()

    def forward(self, stimulus):
        x = self.decode1(stimulus)
        return x


class SimpleAutoEncoder(nn.Module):
    def __init__(self):
        super(SimpleAutoEncoder, self).__init__()
        self.encoder = SimpleEncoder()
        self.decoder = SimpleDecoder()
        
    def forward(self, stimulus):
        x = self.encoder(stimulus)
        x = self.decoder(x)
        return x