import torch
from torch import nn
from torch.distributions import MultivariateNormal, Beta


class ComplexVarAutoEncoder(nn.Module):
    def __init__(self):
        super(ComplexVarAutoEncoder, self).__init__()
        self.encode_mu1 = nn.Linear(220500, 250)
        self.encode_logstd1 = nn.Linear(220500, 250)
        self.encode_mu2 = nn.Linear(250, 100)
        self.encode_logstd2 = nn.Linear(250, 100)
        self.encode_mu3 = nn.Linear(100, 50)
        self.encode_logstd3 = nn.Linear(100, 50)

        self.decode1 = nn.Linear(50, 100)
        self.decode2 = nn.Linear(100, 250)
        self.decode3 = nn.Linear(250, 220500)


        
        self.activation_func = nn.ReLU()


    def forward(self, stimulus):
        mu1 = self.encode_mu1(stimulus)
        logstd1 = self.encode_logstd1(stimulus)
        mu2 = self.encode_mu2(mu1)
        logstd2 = self.encode_logstd2(logstd1)
        mu3 = self.encode_mu3(mu2)
        logstd3 = self.encode_logstd3(logstd2)
        logstd3= torch.clamp(logstd3, -20, 2)
        dist = torch.distributions.MultivariateNormal(
            mu3,
            torch.exp(logstd3).unsqueeze(2) * torch.eye(50, device=stimulus.device).expand(mu1.shape[0], 50, 50)
        )

        # rsample performs reparam trick to keep deterministic and enable backprop
        x = torch.tanh(dist.rsample())

        x = self.decode1(x)
        x = self.decode2(x)
        x = self.decode3(x)
        return x
