# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:38:38 2021

@author: Keith

sr is always 22050
data, sr = librosa.load('Q:\\Documents\\TDS SuperUROP\\music-vae\\wav-clips\\lecture-clip-59.wav')
"""

import os
import torch
import torch.nn as nn
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda:0")


class SimpleLinearAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleLinearAutoencoder, self).__init__()
        self.encode1 = nn.Linear(220500, 50)
        self.decode1 = nn.Linear(50, 220500)
        
        self.activation_func = nn.ReLU()

    def forward(self, stimulus):
        x = self.activation_func(self.encode1(stimulus))
        x = self.decode1(x)
        return x


class ComplexLinearAutoencoder(nn.Module):
    def __init__(self):
        super(ComplexLinearAutoencoder, self).__init__()
        self.encode1 = nn.Linear(220500, 250)
        self.encode2 = nn.Linear(250, 100)
        self.encode3 = nn.Linear(100, 50)
        self.decode1 = nn.Linear(50, 100)
        self.decode2 = nn.Linear(100, 250)
        self.decode3 = nn.Linear(250, 220500)
        
        self.activation_func = nn.ReLU()

    def forward(self, stimulus):
        x = self.activation_func(self.encode1(stimulus))
        x = self.activation_func(self.encode2(x))
        x = self.activation_func(self.encode3(x))
        x = self.activation_func(self.decode1(x))
        x = self.activation_func(self.decode2(x))
        x = self.decode3(x)
        return x


class SimpleConvolutionalEncoder(nn.Module):
    def __init__(self):
        super(SimpleConvolutionalEncoder, self).__init__()
        self.encode1 = nn.Conv1d(1, 4, 501, padding=250)
        self.decode1 = nn.Conv1d(4, 1, 501, padding=250)
        
        self.pool = nn.MaxPool1d(10, stride=10, return_indices=True)
        self.unpool = nn.MaxUnpool1d(10, stride=10)
        self.activation_func = nn.ReLU()

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.activation_func(self.encode1(x))
        y1 = x.size()
        x, p1 = self.pool(x)
        x = self.unpool(x, p1, output_size=y1)
        x = self.decode1(x)
        return torch.squeeze(x)


class ComplexConvolutionalEncoder(nn.Module):
    def __init__(self):
        super(ComplexConvolutionalEncoder, self).__init__()
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


class ImageConvolutionalEncoder(nn.Module):
    def __init__(self):
        super(ImageConvolutionalEncoder, self).__init__()
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
        x = torch.unsqueeze(torch.unsqueeze(real, 0), 0)
        x = self.activation_func(self.encode1(x))
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
        
        x = torch.unsqueeze(torch.unsqueeze(imag, 0), 0)
        x = self.activation_func(self.encode1i(x))
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

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, file_name, list_IDs):
        'Initialization'
        self.list_IDs = list_IDs
        self.file_name = file_name

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]
        dirr = 'Q:/Documents/TDS SuperUROP/music-vae/wav-clips' + os.sep + self.file_name

        # Load data and get label
        data, sr = librosa.load(dirr+str(ID)+'.wav')
        return data
    
    
def TestModel(net, dataset, label, b_size):
    trainfunc = Dataset(dataset,range(1,label))
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=b_size, shuffle=False, num_workers=0)
    net.eval()
    lossfunc = nn.MSELoss()
    
    losses = []
    for i, data in enumerate(trainloader, 0):
        inputs = data.to(device)
        outputs = net(inputs)
        loss = lossfunc(outputs, inputs)
        losses.append(loss.item())
    
    print(dataset)
    return sum(losses)/len(losses)


def OptimizeModel(net, dataset, label, epochs, b_size):    
    # Datasets
    trainfunc = Dataset(dataset,range(1,label))
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=b_size, shuffle=True, num_workers=0)
    
    # Optimize and Loss
    optimizer = torch.optim.Adam(net.parameters())
    lossfunc = nn.MSELoss()
    net.train()
    
    # Train
    loss_results = []
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            inputs = data.to(device)
            outputs = net(inputs)
            loss = lossfunc(outputs, inputs)
            loss.backward()
            optimizer.step()
            
        loss_results.append(TestModel(net, dataset, label, b_size))
        net.train()
        print('Epoch')
    print('Finished Training')
    
    return loss_results


def TestFourierModel(net, dataset, label):
    trainfunc = Dataset(dataset,range(1,label))
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=1, shuffle=False, num_workers=0)
    net.eval()
    lossfunc = nn.MSELoss()
    
    losses = []
    for i, data in enumerate(trainloader, 0):
        comp_input = librosa.core.stft(np.array(data[0]))
        real, imag = torch.Tensor(np.real(comp_input)).to(device), torch.Tensor(np.imag(comp_input)).to(device)
        realOUT, imagOUT = net(real, imag)
        loss_real = lossfunc(realOUT, real)
        loss_imag = lossfunc(imagOUT, imag)
        losses.append(loss_real.item() + loss_imag.item())
    
    print(dataset)
    return sum(losses)/len(losses)


def OptimizeFourierModel(net, dataset, label, epochs):    
    # Datasets
    trainfunc = Dataset(dataset,range(1,label))
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=1, shuffle=True, num_workers=0)
    
    # Optimize and Loss
    optimizer = torch.optim.Adam(net.parameters())
    lossfunc = nn.MSELoss()
    net.train()
    
    # Train
    loss_results = []
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            comp_input = librosa.core.stft(np.array(data[0]))
            real, imag = torch.Tensor(np.real(comp_input)).to(device), torch.Tensor(np.imag(comp_input)).to(device)
            realOUT, imagOUT = net(real, imag)
            loss_real = lossfunc(realOUT, real)
            loss_imag = lossfunc(imagOUT, imag)
            loss_real.backward()
            loss_imag.backward()
            optimizer.step()
            
        loss_results.append(TestFourierModel(net, dataset, label))
        net.train()
        print('Epoch')
    print('Finished Training')
    
    return loss_results


def TestAllModel(net):
    net.eval()
    
    lofi1 = TestModel(net, 'lofi-track-1-clip-', 719, 10)
    lofi2 = TestModel(net, 'lofi-track-2-clip-', 444, 10)
    lec = TestModel(net, 'lecture-clip-', 621, 10)
    jazz = TestModel(net, 'jazz-clip-', 719, 10)
    city = TestModel(net, 'city-sounds-clip-', 719, 10)
    white = TestModel(net, 'white-noise-clip-', 719, 10)
    
    return ['lofi1','lofi2','lec','jazz','city','white'], [lofi1,lofi2,lec,jazz,city,white]


def runSimpleAutoEncode(net, model):
    losses = OptimizeModel(net, 'lofi-track-1-clip-', 719, 10, 30)
    fig, ax = plt.subplots()
    ax.plot(range(1, len(losses)+1),losses)
    ax.set_title('Loss as a function of epoch for '+model)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('MSE Loss')
    plt.savefig('Q:/Documents/TDS SuperUROP/music-vae/models/'+model+'/training_plot.svg')
    plt.show()
    labels, results = TestAllModel(net)
    fig, ax = plt.subplots()
    ax.bar(range(len(results)),results)
    ax.set_title('Loss across data sets for '+model)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Data Set')
    ax.set_ylabel('MSE Loss')
    plt.savefig('Q:/Documents/TDS SuperUROP/music-vae/models/'+model+'/evaluation_plot.svg')
    plt.show()
    torch.save(net.state_dict(), 'Q:/Documents/TDS SuperUROP/music-vae/models/'+model+'/model.pt')
    
    
def TestComplexAllModel(net):
    net.eval()
    
    lofi1 = TestFourierModel(net, 'lofi-track-1-clip-', 2)
    lofi2 = TestFourierModel(net, 'lofi-track-2-clip-', 2)
    lec = TestFourierModel(net, 'lecture-clip-', 2)
    jazz = TestFourierModel(net, 'jazz-clip-', 2)
    city = TestFourierModel(net, 'city-sounds-clip-', 2)
    white = TestFourierModel(net, 'white-noise-clip-', 2)
    
    return ['lofi1','lofi2','lec','jazz','city','white'], [lofi1,lofi2,lec,jazz,city,white]
    
    
def runComplexAutoEncode(net, model):
    losses = OptimizeFourierModel(net, 'lofi-track-1-clip-', 5, 2)
    fig, ax = plt.subplots()
    ax.plot(range(1, len(losses)+1),losses)
    ax.set_title('Loss as a function of epoch for '+model)
    ax.set_xlabel('Training Epoch')
    ax.set_ylabel('MSE Loss')
    plt.savefig('Q:/Documents/TDS SuperUROP/music-vae/models/'+model+'/training_plot.svg')
    plt.show()
    labels, results = TestComplexAllModel(net)
    fig, ax = plt.subplots()
    ax.bar(range(len(results)),results)
    ax.set_title('Loss across data sets for '+model)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels)
    ax.set_xlabel('Data Set')
    ax.set_ylabel('MSE Loss')
    plt.savefig('Q:/Documents/TDS SuperUROP/music-vae/models/'+model+'/evaluation_plot.svg')
    plt.show()
    torch.save(net.state_dict(), 'Q:/Documents/TDS SuperUROP/music-vae/models/'+model+'/model.pt')
    
    
if __name__ == "__main__":
    
    # net = SimpleLinearAutoencoder().to(device)
    # runSimpleAutoEncode(net, 'simpleAutoEncode')
    
    # net = ComplexLinearAutoencoder().to(device)
    # runSimpleAutoEncode(net, 'complexAutoEncode')
    
    # net = SimpleConvolutionalEncoder().to(device)
    # runSimpleAutoEncode(net, 'simpleConvEncode')
    
    # net = ComplexConvolutionalEncoder().to(device)
    # runSimpleAutoEncode(net, 'complexConvEncode')
    
    net = ImageConvolutionalEncoder().to(device)
    runComplexAutoEncode(net, 'imagConvEncode')
    
    print('Done')
    