import torch
from torch import nn
from torch.distributions import MultivariateNormal, Beta


class ComplexVarEncoder(nn.Module):
    def __init__(self):
        super(ComplexVarEncoder, self).__init__()
        self.encode1 = nn.Linear(220500, 250)
        self.encode2 = nn.Linear(250, 100)
        self.encode_mu = nn.Linear(100, 50)
        self.encode_logstd = nn.Linear(100, 50)

        self.activation_func = nn.ReLU()


    def forward(self, stimulus):
        x = self.activation_func(self.encode1(stimulus))
        x = self.activation_func(self.encode2(x))
        mu = self.encode_mu(x)
        logstd = self.encode_logstd(x)
        logstd = torch.clamp(logstd, -20, 2)
        dist = torch.distributions.Normal(mu, torch.exp(logstd))

        return dist


class ComplexVarDecoder(nn.Module):
    def __init__(self):
        super(ComplexVarDecoder, self).__init__()
        self.decode1 = nn.Linear(50, 100)
        self.decode2 = nn.Linear(100, 250)
        self.decode3 = nn.Linear(250, 220500)
        
        self.activation_func = nn.ReLU()


    def forward(self, stimulus):
        x = self.activation_func(self.decode1(stimulus))
        x = self.activation_func(self.decode2(x))
        x = self.decode3(x)
        
        return x


class ComplexVarAutoEncoder(nn.Module):
    def __init__(self):
        super(ComplexVarAutoEncoder, self).__init__()
        self.encoder = ComplexVarEncoder()
        self.decoder = ComplexVarDecoder()
        

    def forward(self, stimulus):
        dist = self.encoder(stimulus)
        # rsample performs reparam trick to keep deterministic and enable backprop
        x = torch.tanh(dist.rsample())
        x = self.decoder(x)
        return x
