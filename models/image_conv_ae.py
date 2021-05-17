import torch
from torch import nn


class ImageConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ImageConvAutoEncoder, self).__init__()
        self.encode1 = nn.Conv2d(1, 8, (500,250))
        self.encode2 = nn.Conv2d(8, 16, (250,100))
        self.encode3 = nn.Conv2d(16, 16, (125,50))
        self.encode4 = nn.Conv2d(16, 16, (75,10))
        self.encode5 = nn.Linear(31600, 1000)
        self.decode5 = nn.Linear(1000, 31600)
        self.decode4 = nn.ConvTranspose2d(16, 16, (75,10), stride=1)
        self.decode3 = nn.ConvTranspose2d(16, 16, (125,50), stride=1)
        self.decode2 = nn.ConvTranspose2d(16, 8, (250,100), stride=1)
        self.decode1 = nn.ConvTranspose2d(8, 1, (500,250), stride=1)
        
        self.encode1i = nn.Conv2d(1, 8, (500,250))
        self.encode2i = nn.Conv2d(8, 16, (250,100))
        self.encode3i = nn.Conv2d(16, 16, (125,50))
        self.encode4i = nn.Conv2d(16, 16, (75,10))
        self.encode5i = nn.Linear(31600, 1000)
        self.decode5i = nn.Linear(1000, 31600)
        self.decode4i = nn.ConvTranspose2d(16, 16, (75,10), stride=1)
        self.decode3i = nn.ConvTranspose2d(16, 16, (125,50), stride=1)
        self.decode2i = nn.ConvTranspose2d(16, 8, (250,100), stride=1)
        self.decode1i = nn.ConvTranspose2d(8, 1, (500,250), stride=1)

        self.activation_func = nn.ReLU()

    def forward(self, real, imag):
        x = self.activation_func(self.encode1(real))
        x = self.activation_func(self.encode2(x))
        x = self.activation_func(self.encode3(x))
        x = self.activation_func(self.encode4(x))
        x = torch.flatten(x, start_dim=1)
        x = self.activation_func(self.encode5(x))
        x = self.activation_func(self.decode5(x))
        x = torch.reshape(x, (1, 16, 79, 25))
        x = self.activation_func(self.decode4(x))
        x = self.activation_func(self.decode3(x))
        x = self.activation_func(self.decode2(x))
        x = self.decode1(x)
        real = torch.squeeze(x)
        
        x = self.activation_func(self.encode1i(imag))
        x = self.activation_func(self.encode2i(x))
        x = self.activation_func(self.encode3i(x))
        x = self.activation_func(self.encode4i(x))
        x = torch.flatten(x, start_dim=1)
        x = self.activation_func(self.encode5i(x))
        x = self.activation_func(self.decode5i(x))
        x = torch.reshape(x, (1, 16, 79, 25))
        x = self.activation_func(self.decode4i(x))
        x = self.activation_func(self.decode3i(x))
        x = self.activation_func(self.decode2i(x))
        x = self.decode1i(x)
        imag = torch.squeeze(x)
        
        return real, imag