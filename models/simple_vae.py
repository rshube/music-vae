import torch
from torch import nn
from torch.distributions import MultivariateNormal, Beta


class SimpleVarAutoEncoder(nn.Module):
    def __init__(self):
        super(SimpleVarAutoEncoder, self).__init__()
        self.encode_mu = nn.Linear(220500, 50)
        self.encode_logstd = nn.Linear(220500, 50)

        self.decode1 = nn.Linear(50, 220500)
        
        self.activation_func = nn.ReLU()


    def forward(self, stimulus):
        mu = self.encode_mu(stimulus)
        logstd = self.encode_logstd(stimulus)
        logstd = torch.clamp(logstd, -20, 2)
        dist = torch.distributions.MultivariateNormal(
            mu,
            torch.exp(logstd).unsqueeze(2) * torch.eye(50, device=stimulus.device).expand(mu.shape[0], 50, 50)
        )

        # rsample performs reparam trick to keep deterministic and enable backprop
        x = torch.tanh(dist.rsample())

        x = self.decode1(x)
        return x
