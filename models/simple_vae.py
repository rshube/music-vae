import torch
from torch import nn
from torch.distributions import MultivariateNormal, Beta


class SimpleVarEncoder(nn.Module):
    def __init__(self):
        super(SimpleVarEncoder, self).__init__()
        self.encode_mu = nn.Linear(220500, 50)
        self.encode_logstd = nn.Linear(220500, 50)
        
        self.activation_func = nn.ReLU()


    def forward(self, stimulus):
        mu = self.encode_mu(stimulus)
        logstd = self.encode_logstd(stimulus)
        logstd = torch.clamp(logstd, -20, 2)
        dist = torch.distributions.Normal(mu, torch.exp(logstd))

        return dist


class SimpleVarDecoder(nn.Module):
    def __init__(self):
        super(SimpleVarDecoder, self).__init__()
        self.decode1= nn.Linear(50, 220500)
        

    def forward(self, stimulus):
        x = self.decode1(stimulus)

        return x


class SimpleVarAutoEncoder(nn.Module):
    def __init__(self):
        super(SimpleVarAutoEncoder, self).__init__()
        self.encoder = SimpleVarEncoder()
        self.decoder = SimpleVarDecoder()


    def forward(self, stimulus):
        dist = self.encoder(stimulus)
        # rsample performs reparam trick to keep deterministic and enable backprop
        x = torch.tanh(dist.rsample())
        x = self.decoder(x)
        return x

