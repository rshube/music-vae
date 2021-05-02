# -*- coding: utf-8 -*-
"""
Created on Sun May  2 15:38:38 2021

@author: Keith
"""

import os
import torch
import torch.nn as nn
import librosa
import matplotlib.pyplot as plt

device = torch.device("cuda:0")


class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super(SimpleAutoencoder, self).__init__()
        self.encode1 = nn.Linear(220500, 300)
        self.encode2 = nn.Linear(300,50)
        self.encode3 = nn.Linear(50,25)
        self.decode1 = nn.Linear(25,50)
        self.decode2 = nn.Linear(50,300)
        self.decode3 = nn.Linear(300, 220500)
        
        self.activation_func = nn.ReLU()

    def forward(self, stimulus):
        x = self.activation_func(self.encode1(stimulus))
        x = self.activation_func(self.encode2(x))
        x = self.activation_func(self.encode3(x))
        x = self.activation_func(self.decode1(x))
        x = self.activation_func(self.decode2(x))
        x = self.decode3(x)
        return x


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
        dirr = 'Q:\Documents\TDS SuperUROP\\music-vae\\'+self.file_name + os.sep

        # Load data and get label
        data, sr = librosa.load(dirr+'\lofi-track-1-clip-'+str(ID)+'.mp3')
        return data
    
    
def OptimizeModel(net, dataset, label, epochs):    
    # Datasets
    trainfunc = Dataset(dataset,range(1,label))
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=30, shuffle=True, num_workers=0)
    
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
            
            loss_results.append(loss.item())
    
    plt.plot(loss_results)
    print('Finished Training')
    
    
if __name__ == "__main__":
    net = SimpleAutoencoder().to(device)
    OptimizeModel(net, 'clips', 444, 10)
    