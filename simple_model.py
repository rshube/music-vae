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
from datetime import datetime

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

    
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
    trainfunc = Dataset(dataset,range(1, label))
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
        print(f'Epoch {epoch}\tEvaluation Loss: {loss_results[-1]}')
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
        print(f'Epoch {epoch}')
    print('Finished Training')
    
    return loss_results


def TestAllModel(net):
    net.eval()
    
    lofi1 = TestModel(net, 'lofi-track-1-clip-', 720, 10)
    lofi2 = TestModel(net, 'lofi-track-2-clip-', 444, 10)
    lec = TestModel(net, 'lecture-clip-', 621, 10)
    jazz = TestModel(net, 'jazz-clip-', 720, 10)
    city = TestModel(net, 'city-sounds-clip-', 720, 10)
    white = TestModel(net, 'white-noise-clip-', 720, 10)
    
    return ['lofi1','lofi2','lec','jazz','city','white'], [lofi1,lofi2,lec,jazz,city,white]


def TestAllFourierModel(net):
    net.eval()
    
    lofi1 = TestFourierModel(net, 'lofi-track-1-clip-', 2)
    lofi2 = TestFourierModel(net, 'lofi-track-2-clip-', 2)
    lec = TestFourierModel(net, 'lecture-clip-', 2)
    jazz = TestFourierModel(net, 'jazz-clip-', 2)
    city = TestFourierModel(net, 'city-sounds-clip-', 2)
    white = TestFourierModel(net, 'white-noise-clip-', 2)
    
    return ['lofi1','lofi2','lec','jazz','city','white'], [lofi1,lofi2,lec,jazz,city,white]


    
if __name__ == "__main__":
    
    # net = SimpleLinearAutoencoder().to(device)
    # runAutoEncode(net, 'simpleAutoEncode')

    net = SimpleLinearVariationalAutoencoder().to(device)
    runAutoEncode(net, 'simpleVarAutoEncode')
    
    # net = ComplexLinearAutoencoder().to(device)
    # runAutoEncode(net, 'complexAutoEncode')
    
    # net = SimpleConvolutionalEncoder().to(device)
    # runAutoEncode(net, 'simpleConvEncode')
    
    # net = ComplexConvolutionalEncoder().to(device)
    # runAutoEncode(net, 'complexConvEncode')
    
    # net = ImageConvolutionalEncoder().to(device)
    # runAutoEncode(net, 'imagConvEncode', fourier=True)
    
    print('Done')
    