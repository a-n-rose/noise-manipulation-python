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

import matplotlib.pyplot as plt
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
s_wav = "./German-speech-sample-female.wav"
s,sr = librosa.load(s_wav, sr= 16000,duration=2)
# noise samples (make same length as recorded sample)
n_wav = "./bus.wav"
n, sr = librosa.load(n_wav, sr=16000,duration=2)

s, n = match_lengths(s,n)


# original signal with noise
x = s+n

#stft of all
s_stft = librosa.stft(s)
n_stft = librosa.stft(n)
x_stft = librosa.stft(x)

s_pow = np.abs(s_stft)**2
x_pow = np.abs(x_stft)**2


N = len(x)
M = 256  # length of Wiener filter

# estimate (cross) PSDs using Welch technique
f, Pxx = sig.csd(x_pow, x_pow)
f, Psx = sig.csd(s_pow, x_pow)
# compute Wiener filter
H = Psx/Pxx
H = H * np.exp(-1j*2*np.pi/len(H)*np.arange(len(H))*(len(H)//2))  # shift for causal filter
h = np.fft.irfft(H)
# apply Wiener filter to observation
y = np.convolve(x, h, mode='same')

# plot (cross) PSDs
Om = np.linspace(0, np.pi, num=len(H))



plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(Om, 20*np.log10(np.abs(.5*Pxx)), label=r'$| \Phi_{xx}(e^{j \Omega}) |$ in dB')
plt.plot(Om, 20*np.log10(np.abs(.5*Psx)), label=r'$| \Phi_{sx}(e^{j \Omega}) |$ in dB')
plt.title('(Cross) PSDs')
plt.xlabel(r'$\Omega$')
plt.legend()
plt.axis([0, np.pi, -60, 40])
plt.grid()

# plot transfer function of Wiener filter
plt.subplot(122)
plt.plot(Om, 20*np.log10(np.abs(H)))
plt.title('Transfer function of Wiener filter')
plt.xlabel(r'$\Omega$')
plt.ylabel(r'$| H(e^{j \Omega}) |$ in dB')
plt.axis([0, np.pi, -150, 3])
plt.grid()
plt.tight_layout()

# plot signals
idx = np.arange(500, 600)
plt.figure(figsize=(10, 4))
plt.plot(idx, x[idx], label=r'observed signal $x[k]$')
plt.plot(idx, s[idx], label=r'original signal $s[k]$')
plt.plot(idx, y[idx], label=r'estimated signal $y[k]$')
plt.title('Signals')
plt.xlabel(r'$k$')
plt.axis([idx[0], idx[-1], -1.5, 1.5])
plt.legend()
plt.grid()
plt.show()

sf.write("wiener_test.wav",y,1600)
