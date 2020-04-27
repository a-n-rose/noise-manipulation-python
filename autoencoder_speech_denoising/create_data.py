# create clean and noisy speech files

# For PySoundTool
# TODO add error handling for too large of sound files
# TODO add function for getting len in seconds of wavefile
# TODO allow new directories to be made
# TODO allow for random start to noise - perhaps start from beginning if close to end?
# TODO allow to add sounds from data not filename
# TODO remove delay from add sound to signal
# TODO remove clicks from ending of wavefiles
# TODO add function for finding signal to noise ratio
# TODO separate feature extraction from visualization

import glob
import numpy as np
from scipy.io.wavfile import write
import random
import os
import pysoundtool.explore_sound as exsound 
import pysoundtool.soundprep as soundprep
import pysoundtool as pyst 


def get_time_sec(audiodata, sr):
    time_sec = len(audiodata)/sr
    return time_sec

path2cleanspeech = '/home/airos/Projects/Data/CDBook_SpeechEnhancement/Databases/Speech/IEEE_corpus/wideband/'

path2noise = '/home/airos/Projects/Data/CDBook_SpeechEnhancement/Databases/Noise Recordings/'

path2save_clean = '/home/airos/Projects/Data/clean_speech/'
path2save_noisy = '/home/airos/Projects/Data/noisy_speech/'
# need to save README for these data

cleanspeechwaves = glob.glob(path2cleanspeech+'*.wav')
noisewaves = glob.glob(path2noise+'*.wav')

num_noises = len(noisewaves)

for directory in [path2save_clean,
                 path2save_noisy]:
    if not os.path.exists(directory):
        os.mkdir(directory)

scale = 0.3
for wavefile in cleanspeechwaves:
    rand_noise_id = random.choice(
        range(num_noises))
    noise = noisewaves[rand_noise_id]
    
    # to save in noisy speech wavefile
    speech_stem = os.path.splitext(
        os.path.basename(wavefile))[0]
    noise_stem = os.path.splitext(
        os.path.basename(noise))[0]
    
    noise_data, sr = soundprep.loadsound(
        noise)
    speech_data, sr2 = soundprep.loadsound(
        wavefile)
    speech_seconds = get_time_sec(
        speech_data, sr2)
    noisyspeech, sr = soundprep.add_sound_to_signal(
        wavefile, noise, scale = scale, delay_target_sec=0, total_len_sec = speech_seconds
        )
    # TODO: allow for random sectioning of noise
    
    write(
        '{}{}__{}_scale{}.wav'.format(path2save_noisy,
                              speech_stem,noise_stem,
                              scale), 
        sr, 
        noisyspeech)
    write(path2save_clean+speech_stem+'.wav', sr, speech_data)
