import torch
from torch import nn
import numpy as np
import librosa

from utils import Dataset, DATA_PATH

def TestModel(model, dataset, num_clips, b_size):
    trainfunc = Dataset(dataset,range(1,num_clips))
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=b_size, shuffle=False, num_workers=0)
    model.eval()
    lossfunc = nn.MSELoss()
    
    losses = []
    for i, data in enumerate(trainloader, 0):
        inputs = data.to(model.device)
        outputs = model(inputs)
        loss = lossfunc(outputs, inputs)
        losses.append(loss.item())
    
    print(dataset)
    return sum(losses)/len(losses)


def TestFourierModel(model, dataset, num_clips):
    trainfunc = Dataset(dataset,range(1,num_clips))
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    lossfunc = nn.MSELoss()
    
    losses = []
    for i, data in enumerate(trainloader, 0):
        comp_input = librosa.core.stft(np.array(data[0]))
        real, imag = torch.Tensor(np.real(comp_input)).to(model.device), torch.Tensor(np.imag(comp_input)).to(model.device)
        realOUT, imagOUT = model(real, imag)
        loss_real = lossfunc(realOUT, real)
        loss_imag = lossfunc(imagOUT, imag)
        losses.append(loss_real.item() + loss_imag.item())
    
    print(dataset)
    return sum(losses)/len(losses)


def OptimizeModel(args, model, dataset, num_clips):    
    # Datasets
    trainfunc = Dataset(dataset,range(1, num_clips))
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Optimize and Loss
    optimizer = torch.optim.Adam(model.parameters())
    lossfunc = nn.MSELoss()
    model.train()
    
    # Train
    loss_results = []
    for epoch in range(args.num_epochs):
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            inputs = data.to(model.device)
            outputs = model(inputs)
            loss = lossfunc(outputs, inputs)
            loss.backward()
            optimizer.step()
            
        loss_results.append(TestModel(model, DATA_PATH, dataset, num_clips, args.batch_size))
        model.train()
        print(f'Epoch {epoch}\tEvaluation Loss: {loss_results[-1]}')
    print('Finished Training')
    
    return loss_results


def OptimizeFourierModel(model, dataset, num_clips, epochs):    
    # Datasets
    trainfunc = Dataset(dataset,range(1,num_clips))
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=1, shuffle=True, num_workers=0)
    
    # Optimize and Loss
    optimizer = torch.optim.Adam(model.parameters())
    lossfunc = nn.MSELoss()
    model.train()
    
    # Train
    loss_results = []
    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            optimizer.zero_grad()
            comp_input = librosa.core.stft(np.array(data[0]))
            real, imag = torch.Tensor(np.real(comp_input)).to(model.device), torch.Tensor(np.imag(comp_input)).to(model.device)
            realOUT, imagOUT = model(real, imag)
            loss_real = lossfunc(realOUT, real)
            loss_imag = lossfunc(imagOUT, imag)
            loss_real.backward()
            loss_imag.backward()
            optimizer.step()
            
        loss_results.append(TestFourierModel(model, DATA_PATH, dataset, num_clips))
        model.train()
        print(f'Epoch {epoch}')
    print('Finished Training')
    
    return loss_results