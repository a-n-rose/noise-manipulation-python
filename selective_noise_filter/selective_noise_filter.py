'''
Little example of selective filtering

'''
import os
import sys
import datetime

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa

import prep_noise as pn


def get_date():
    '''
    This creates a string of the day, hour, minute and second
    I use this to make folder names unique
    
    For the files themselves, I generate genuinely unique names (i.e. name001.csv, name002.csv, etc.)
    '''
    time = datetime.datetime.now()
    time_str = "{}y{}m{}d{}h{}m{}s".format(time.year,time.month,time.day,time.hour,time.minute,time.second)
    return(time_str)

def record(duration,fs,channels):
    print("Recording noise for {} seconds...".format(duration))
    sound = sd.rec(int(duration * fs), samplerate=fs, channels=2)
    sd.wait()
    return sound


def main():
    sr = 16000

    timestamp = get_date()
    print("Press Enter to record a sound you're tired of hearing (e.g. an air conditioner, typing keys, etc.)")
    ready = input()
    if ready == "":
        noise = record(5,sr,2)
    else:
        print("Bye!")
        sys.exit()
        
    print("\nPlease enter the name of this noise:")
    noise_name = input()
    
    print("\nGreat! \n\nNow the program will record your environment for 20 seconds.\nPlay some nice music, program on your computer, or have a nice chat with someone. The nasty noise from before will be (hopefully) removed.")
    print("\nPress Enter to start:")
    ready = input()
    if ready == "":
        environment = record(20,sr,2)
    else:
        print("Bye!")
        sys.exit()
        
    recordings_folder = "recordings/{}_{}".format(noise_name,timestamp)
    if not os.path.exists(recordings_folder):
        os.makedirs(recordings_folder)
        
    noise_wave = "./{}/noise.wav".format(recordings_folder)
    sf.write(noise_wave,noise,sr)
    env_wave = "./{}/environment.wav".format(recordings_folder)
    sf.write(env_wave,environment,sr)
    
    n, sr = librosa.load(noise_wave,sr=sr)
    env, sr = librosa.load(env_wave,sr=sr)
    env_rn = pn.rednoise(env,n,sr,5)
    
    env_rn_wave = "./{}/environment_red_noise.wav".format(recordings_folder)
    sf.write(env_rn_wave,env_rn,sr)
    
    print("\n\nCheck the new recordings in the folder: {}\n\n".format(recordings_folder))
    
    
    return None


if __name__=="__main__":
    main()
