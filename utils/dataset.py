import os
from utils.consts import DATA_PATH
import torch
import librosa


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
        dirr = os.path.join(DATA_PATH, f'wav-clips/{self.file_name}')

        # Load data and get label
        data, sr = librosa.load(f'{dirr}{ID}.wav')
        return data