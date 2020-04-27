import os
from keras.models import load_model
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Conv2DTranspose
from keras.constraints import max_norm
import matplotlib.pyplot as plt
import numpy as np
import math
import glob
import datetime
from python_speech_features import mfcc, logfbank
import pysoundtool.explore_sound as exsound 
import pysoundtool.soundprep as soundprep
import pysoundtool as pyst 
from extract_data import visualize_feats

model_filename  = 'autoencoder1_epochs50_4m27d17h58m51s.h5'


# Load data
noisy_data = np.load('./sample_data/noisy_speech_20files_batchsize30_numframes12.npy')
pure_data = np.load('./sample_data/clean_speech_20files_batchsize30_numframes12.npy')


# normalize data
for i in range(len(noisy_data)):
    noisy_sample = noisy_data[i]
    pure_sample = pure_data[i]
    noisy_data[i] = (noisy_sample - np.min(noisy_sample)) / (np.max(noisy_sample) - np.min(noisy_sample))
    pure_data[i] = (pure_sample - np.min(pure_sample)) / (np.max(pure_sample) - np.min(pure_sample))
    
#reshape to mix samples and batchsizes:
assert noisy_data.shape == pure_data.shape
noisy_input = noisy_data.reshape((noisy_data.shape[0] * noisy_data.shape[1],
                                  noisy_data.shape[2], noisy_data.shape[3],1))
pure_input = pure_data.reshape((pure_data.shape[0] * pure_data.shape[1],
                                  pure_data.shape[2], pure_data.shape[3],1))

    
num_feats, num_frames = noisy_data.shape[-1], noisy_data.shape[2]
batch_size = noisy_data.shape[1]
train_test_split = 0.3


modelname = model_filename.split('.')[0] 
image_folder = './images/'
models_folder = './models/'

# Train/test split
percentage_training = math.floor((1 - train_test_split) * len(noisy_input))
noisy_input, noisy_input_test = noisy_input[:percentage_training], noisy_input[percentage_training:]
pure_input, pure_input_test = pure_input[:percentage_training], pure_input[percentage_training:]




autoencoder = load_model(models_folder + model_filename)

# Generate reconstructions
num_reconstructions = 20
samples = noisy_input_test[:num_reconstructions*batch_size]
if len(samples) < num_reconstructions*batch_size:
    print('Not sufficient data for {} reconstructions.'.format(
        num_reconstructions))
    num_reconstructions = len(samples)//batch_size
    print('Reconstructions limited to {}.'.format(num_reconstructions))
reconstructions = autoencoder.predict(samples, batch_size=batch_size)
if percentage_training < round((1 - train_test_split) * len(noisy_input),0):
    index_adjustment = 1
else:
    index_adjustment = 0

# Plot reconstructions
for i in np.arange(0, num_reconstructions):
  # Prediction index
  prediction_index = i + (percentage_training//batch_size) + index_adjustment
  # Get the sample and the reconstruction
  original = noisy_data[prediction_index]
  pure = pure_data[prediction_index]
  reconstruction = np.array(reconstructions[i*batch_size:batch_size+i*batch_size]).reshape((batch_size * num_frames, num_feats)) 
  visualize_feats(
    original.reshape((
        original.shape[0]*original.shape[1], 
        original.shape[2])),
    feature_type='fbank', 
    save_pic=True, 
    name4pic='{}index{}_noisy_signal'.format(image_folder,prediction_index))
  visualize_feats(
    pure.reshape((
        pure.shape[0]*pure.shape[1], 
        pure.shape[2])),
    feature_type='fbank', 
    save_pic=True, 
    name4pic='{}index{}_clean_signal'.format(image_folder,prediction_index))
  visualize_feats(reconstruction, feature_type='fbank', save_pic=True,
                  name4pic='{}index{}_{}'.format(image_folder,prediction_index,modelname))

