import torch
from torch import nn
import numpy as np

from .evaluating import TestModel
from utils import Dataset, TRAINING_DATASET


def TrainModel(args, model, num_clips, fourier=False):    
    # Datasets
    trainfunc = Dataset(TRAINING_DATASET,range(1, num_clips))
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Optimize and Loss
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    lossfunc = nn.MSELoss()
    model.train()
    
    # Train
    eval_results = []
    for epoch in range(args.num_epochs):
        for i, data in enumerate(trainloader):
            optimizer.zero_grad()
            if not fourier:
                inputs = data.to(args.device)
                outputs = model(inputs)
                loss = lossfunc(outputs, inputs)
                loss.backward()
            else:
                comp_input = torch.stft(data, n_fft=2048, window=torch.hann_window(2048), return_complex=True)
                real, imag = comp_input.real.unsqueeze(0).to(args.device), comp_input.imag.unsqueeze(0).to(args.device)
                realOUT, imagOUT = model(real, imag)
                loss_real = lossfunc(realOUT, real)
                loss_imag = lossfunc(imagOUT, imag)
                loss_real.backward()
                loss_imag.backward()
            optimizer.step()
            
        eval_results.append(TestModel(args, model, num_clips, fourier=fourier))
        print(f'[Epoch {epoch}]\tEvaluation Loss: {eval_results[-1]}')
    print('Finished Training')
    
    return eval_results