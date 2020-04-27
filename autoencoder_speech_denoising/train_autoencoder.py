# started from: https://www.machinecurve.com/index.php/2019/12/19/creating-a-signal-noise-removal-autoencoder-with-keras/#creating-the-autoencoder

# incorporated code from: https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f

# TODO reduce learning rate of validation loss does not decrease 
# TODO transform fbank features to realtime audio
# TODO collect more data: google dataset search
# TODO also log settings of models saved under modelname.. learning rate and such

import os
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


def get_date():
    time = datetime.datetime.now()
    time_str = "{}m{}d{}h{}m{}s".format(time.month,time.day,time.hour,time.minute,time.second)
    return(time_str)



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



# Model configuration
model_version = 1 # 1, 2, 
num_feats, num_frames = noisy_data.shape[-1], noisy_data.shape[2]
batch_size = noisy_data.shape[1]
input_shape = (num_frames, num_feats, 1)
no_epochs = 50
train_test_split = 0.3
validation_split = 0.2
verbosity = 1
max_norm_value = 2.0

timestamp = get_date()
modelname = 'autoencoder{}_epochs{}_{}'.format(model_version,no_epochs,timestamp)
image_folder = './images/'
log_folder = './model_logs/'
models_folder = './models/'

for directory in [image_folder, log_folder, models_folder]:
    if not os.path.exists(directory):
        os.mkdir(directory)

# Time comparison between models
#Train on 12095 samples, validate on 3024 samples
# Model version 2, 1 epochs, total time:  9.066
# Model version 1, 1 epochs, total time:  48.523
# autoencoder2_epochs50_4m26d20h38m41s total time:  154.229 seconds; 17 epochs; val_loss: 0.0146 autoencoder1_epochs50_4m26d20h42m1s total time:  1811.543; 37 epochs; val_loss: 0.5932

# model 1 (with 37 epochs) does seem to make better pictures
# try model 2 with lower learning rate
#autoencoder2_epochs50_4m26d22h5m3s total time:  162.427; 18 epochs; val_loss: 0.0161



# Train/test split
percentage_training = math.floor((1 - train_test_split) * len(noisy_input))
noisy_input, noisy_input_test = noisy_input[:percentage_training], noisy_input[percentage_training:]
pure_input, pure_input_test = pure_input[:percentage_training], pure_input[percentage_training:]

if model_version == 1:
    # Create the model - perhaps for raw signals
    autoencoder = Sequential()
    autoencoder.add(Conv2D(128, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))
    autoencoder.add(Conv2D(32, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
    autoencoder.add(Conv2DTranspose(32, kernel_size=(3,3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
    autoencoder.add(Conv2DTranspose(128, kernel_size=(3,3), kernel_constraint=max_norm(max_norm_value), activation='relu', kernel_initializer='he_uniform'))
    autoencoder.add(Conv2D(1, kernel_size=(3, 3), kernel_constraint=max_norm(max_norm_value), activation='sigmoid', padding='same'))
    # Compile and fit data
    adm = keras.optimizers.Adam(learning_rate=0.0001)
    autoencoder.compile(optimizer=adm, loss='binary_crossentropy')

elif model_version == 2:
    from keras.models import Model
    from keras.layers import Input, Dense, UpSampling2D, MaxPooling2D

    x = Input(shape=input_shape)

    # Encoder
    conv1_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    pool1 = MaxPooling2D((2, 2), padding='same')(conv1_1)
    conv1_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(pool1)
    h = MaxPooling2D((2, 2), padding='same')(conv1_2)

    # Decoder
    conv2_1 = Conv2D(32, (3, 3), activation='relu', padding='same')(h)
    up1 = UpSampling2D((2, 2))(conv2_1)
    conv2_2 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
    up2 = UpSampling2D((2, 2))(conv2_2)
    r = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2)

    autoencoder = Model(input=x, output=r)
    adm = keras.optimizers.Adam(learning_rate=0.0001)
    autoencoder.compile(optimizer=adm, loss='mse')

from keras.callbacks import EarlyStopping,ReduceLROnPlateau,CSVLogger,ModelCheckpoint

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=5)
csv_logging = CSVLogger(filename='{}{}_log.csv'.format(log_folder, modelname))
checkpoint_callback = ModelCheckpoint(models_folder+modelname+'.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')


import time
start = time.time()
autoencoder.summary()
autoencoder.fit(noisy_input, pure_input,
                epochs=no_epochs,
                batch_size=batch_size,
                callbacks=[early_stopping_callback, checkpoint_callback, csv_logging],
                validation_split=validation_split)
end = time.time()
print('{} total time: {} seconds'.format(modelname,round(end-start,3)))

