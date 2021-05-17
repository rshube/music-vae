import torch
import torch.nn as nn
import numpy as np

from utils import Dataset, TRAINING_DATASET

def TestModel(args, model, dataset, num_clips, fourier=False):
    trainfunc = Dataset(dataset,range(1,num_clips))
    trainloader = torch.utils.data.DataLoader(trainfunc, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model.eval()
    lossfunc = nn.MSELoss()
    
    losses = []
    for i, data in enumerate(trainloader):
        if not fourier:
            inputs = data.to(args.device)
            outputs = model(inputs)
            loss = lossfunc(outputs, inputs)
            losses.append(loss.item())
        else:
            comp_input = torch.stft(data.squeeze(), n_fft=2048, window=torch.hann_window(2048), return_complex=True)
            real, imag = comp_input.real.to(args.device), comp_input.imag.to(args.device)
            real, imag = real[None, None], imag[None, None]   # unsqueeze twice in 0th dim
            realOUT, imagOUT = model(real, imag)
            loss_real = lossfunc(realOUT, real)
            loss_imag = lossfunc(imagOUT, imag)
            losses.append(loss_real.item() + loss_imag.item())
    
    return np.mean(losses)



def TestAllModel(args, model, num_clips_dict, fourier=False):
    model.eval()
    
    lofi1 = TestModel(args, model, 'lofi-track-1-clip-', num_clips=num_clips_dict['lofi-track-1-clip-'], fourier=fourier)
    lofi2 = TestModel(args, model, 'lofi-track-2-clip-', num_clips=num_clips_dict['lofi-track-2-clip-'], fourier=fourier)
    lec = TestModel(args, model, 'lecture-clip-', num_clips=num_clips_dict['lecture-clip-'], fourier=fourier)
    jazz = TestModel(args, model, 'jazz-clip-', num_clips=num_clips_dict['jazz-clip-'], fourier=fourier)
    city = TestModel(args, model, 'city-sounds-clip-', num_clips=num_clips_dict['city-sounds-clip-'], fourier=fourier)
    white = TestModel(args, model, 'white-noise-clip-', num_clips=num_clips_dict['white-noise-clip-'], fourier=fourier)
    
    return ['lofi1','lofi2','lec','jazz','city','white'], [lofi1,lofi2,lec,jazz,city,white]