from torch import nn


class SimpleAutoEncoder(nn.Module):
    def __init__(self):
        super(SimpleAutoEncoder, self).__init__()
        self.encode1 = nn.Linear(220500, 50)
        self.decode1 = nn.Linear(50, 220500)
        
        self.activation_func = nn.ReLU()

    def forward(self, stimulus):
        x = self.activation_func(self.encode1(stimulus))
        x = self.decode1(x)
        return x