# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 22:25:40 2021

@author: murra
"""
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# data, sr = librosa.load('example.mp3')
# x = data[:10*sr]
# plt.plot(x)
# np.save('example.npy', x)

y = np.load('example.npy')
A = librosa.stft(y)
S = np.imag(A)

fig, ax = plt.subplots()
img = librosa.display.specshow(librosa.amplitude_to_db(S,ref=np.max),y_axis='log', x_axis='time', ax=ax)
ax.set_title('Power spectrogram')
fig.colorbar(img, ax=ax, format="%+2.0f dB")