# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:38:38 2021

@author: Keith
"""

import os
import torch
import torch.nn as nn
import librosa
import soundfile as sf
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
        self.encode1 = nn.Conv1d(1, 4, 501, padding=255)
        self.decode1 = nn.ConvTranspose1d(4, 1, 501, padding=255)
        
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
        self.encode1 = nn.Conv1d(1, 4, 501, padding=255)
        self.encode2 = nn.Conv1d(4, 8, 251, padding=125)
        self.encode3 = nn.Conv1d(8, 16, 125, padding=62)
        self.decode1 = nn.ConvTranspose1d(16, 8, 125, padding=62)
        self.decode2 = nn.ConvTranspose1d(8, 4, 252, padding=125)
        self.decode3 = nn.ConvTranspose1d(4, 1, 501, padding=255)
        
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
        x = self.unpool(x, p2)
        x = self.activation_func(self.decode2(x))
        x = self.unpool(x, p1, output_size=y1)
        x = self.decode3(x)
        return torch.squeeze(x)


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
    
    
if __name__ == "__main__":
    
    net = SimpleLinearAutoencoder().to(device)
    runSimpleAutoEncode(net, 'simpleAutoEncode')
    
    net = ComplexLinearAutoencoder().to(device)
    runSimpleAutoEncode(net, 'complexAutoEncode')
    
    net = SimpleConvolutionalEncoder().to(device)
    runSimpleAutoEncode(net, 'simpleConvEncode')
    
    net = ComplexConvolutionalEncoder().to(device)
    runSimpleAutoEncode(net, 'complexConvEncode')
    
    
    
    