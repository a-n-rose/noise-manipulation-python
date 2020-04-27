'''
This is an approximation of Wiener filter noise filtering.

Helpful tutorial:
https://dsp-nbsphinx.readthedocs.io/en/nbsphinx-experiment/random_signals_LTI_systems/wiener_filter.html

'''

import numpy as np
import librosa
import math
import random
import soundfile as sf


#import matplotlib.pyplot as plt
import scipy.signal as sig



def match_lengths(target, noise):
    if len(target) != len(noise):
        smaller = np.argmin([len(target),len(noise)])
        diff = abs(len(target)-len(noise))
        if smaller == 0:
            target = list(target)
            for i in range(diff):
                target.append(target[i])
            target = np.array(target)
        else:
            noise = list(noise)
            for i in range(diff):
                noise.append(noise[i])
            noise = np.array(noise)
    return target, noise



# orignal signal
s_wav = "//home/airos/Data/sound/aislyn_audacity_reducednoise.wav"
s,sr = librosa.load(s_wav, sr= 16000,duration=10)
# noise samples (make same length as recorded sample)
n_wav = "//home/airos/Data/sound/noise_keyboard.wav"
n, sr = librosa.load(n_wav, sr=16000,duration=2)

s, n = match_lengths(s,n)
x = s+n

f, Pxx = sig.csd(x,x,fs=16000,scaling='density')
f, Psx = sig.csd(s,x,fs=16000,scaling='density')

#calculate Wiener filter
H = Psx/Pxx
H = H * np.exp(-1j*2*np.pi/len(H)*np.arange(len(H))*(len(H)//2))  # shift for causal filter
h = np.fft.irfft(H)
#apply Wiener filter to observation
y = np.convolve(x,h, mode='same')

print(y.shape)

 


sf.write("wiener_test_n_nr.wav",y,16000)
sf.write("wiener_test_n.wav",x,16000)
