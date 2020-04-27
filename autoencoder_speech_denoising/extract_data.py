import matplotlib.pyplot as plt
import numpy as np
import math
import glob
from python_speech_features import mfcc, logfbank
import pysoundtool.explore_sound as exsound 
import pysoundtool.soundprep as soundprep
import pysoundtool as pyst 




def visualize_feats(feature_matrix, feature_type, save_pic=False, name4pic=None):
    '''Visualize feature extraction; frames on x axis, features on y axis
    
    Parameters
    ----------
    feature_matrix : numpy.ndarray
        Matrix of feeatures.
    feature_type : str
        Either 'mfcc' or 'fbank' features. MFCC: mel frequency cepstral
        coefficients; FBANK: mel-log filterbank energies (default 'fbank')
    '''
    if 'fbank' in feature_type:
        axis_feature_label = 'Mel Filters'
    elif 'mfcc' in feature_type:
        axis_feature_label = 'Mel Freq Cepstral Coefficients'

    plt.clf()
    plt.pcolormesh(feature_matrix.T)
    plt.xlabel('Frames')
    plt.ylabel('Num {}'.format(axis_feature_label))
    plt.title('{} Features'.format(feature_type.upper()))
    if save_pic:
        outputname = name4pic or 'visualize{}feats'.format(feature_type.upper())
        plt.savefig('{}.png'.format(outputname))
    else:
        plt.show()

def check_noisy_clean_match(cleanfilename, noisyfilename):
    import os
    clean = os.path.splitext(os.path.basename(cleanfilename))[0]
    noisy = os.path.splitext(os.path.basename(noisyfilename))[0]
    if clean in noisy:
        return True
    else:
        print('{} is not in {}.'.format(clean, noisy))
        return False
    
def check_length_match(cleanfilename, noisyfilename):   
    clean, sr1 = soundprep.loadsound(cleanfilename)
    noisy, sr2 = soundprep.loadsound(noisyfilename)
    assert sr1 == sr2
    if len(clean) != len(noisy):
        print('length of clean speech: ', len(clean))
        print('length of noisy speech: ', len(noisy))
    else:
        print('length matches!')
    return None

# window of 25ms? 
# TODO: look into research about this this
# Lu, X., Tsao, Y., Matsuda, S., Hori, C.(2013) Speech Enhancement Based on Deep Denoising Autoencoder
# 16ms windows with 8 ms window shift
# 40 fbank mel frequncy power spectrum
# SNR 0, 5, 10dB
# 350 utterances (training)
# 50 utterances (testing)
# 2 types of noise: factory and car noise signals
# size of spectral patches tried: 3, 7, 11 frames and dimensions to autoencoder: 120,  280, 440 respectively
# 11 frame patches used
# (11, 440) shape?  11 * 40 fbank = 440 


# 11 frames: 16 ms windows, w 8 ms overlap
# 8 * 11 + 8 = 96ms
# batch size of 10 is 960ms, almost 1 second

def get_feats_directory(wavefile_directory, feature_type='fbank',
                        win_size_ms = 16, 
                        win_shift_ms = 8,num_features=40, 
                        frames_per_sample = 11,batch_size=10):
    '''sets up wavefiles, feed to other functions, saves features.
    batch_size limits amount of time per audio file.
    
    Parameters
    ----------
    wavefile_directory : str
        Directory where sound files are stored.
    
    feature_type : str
        Options: 'fbank' (coming soon: 'mfcc', 'stft', 'signal')
        
    Returns
    -------
    feat_matrix : np.ndarray 
        Shape (num_audiofiles, batch_size, frames_per_sample, num_features)
    '''
    audiopaths = get_audiopaths(wavefile_directory)
    feat_matrix = np.zeros((len(audiopaths),
                            batch_size * frames_per_sample,
                            num_features))
    for i, sound in enumerate(audiopaths):
        feats = get_feats(sound, feature_type, win_size_ms = win_size_ms,
            win_shift_ms = win_shift_ms,num_filters=num_features)[:batch_size*frames_per_sample,:]
        feat_matrix[i,:feats.shape[0],:] += feats 
    feat_matrix = feat_matrix.reshape((feat_matrix.shape[0],
                                       batch_size,
                                       frames_per_sample,
                                       num_features))
    return feat_matrix

def get_audiopaths(directory):
    '''Collects .wav files in a directory and sorts them. (Currently, only
        .wav files are processed)
        TODO: use pathlib (safe for Linux for now)
    '''
    import glob
    if directory[-1] != '/':
        directory += '/'
    audiopaths = sorted(glob.glob(directory+'*.wav'))
    return audiopaths

def get_feats(sound, features='fbank', win_size_ms = 20, \
    win_shift_ms = 10,num_filters=40,num_mfcc=40, samplerate=None):
    '''Feature extraction depending on set parameters; frames on y axis, features x axis
    
    Parameters
    ----------
    sound : str or numpy.ndarray
        If str, wavfile (must be compatible with scipy.io.wavfile). Otherwise 
        the samples of the sound data. Note: in the latter case, `samplerate`
        must be declared.
    features : str
        Either 'mfcc' or 'fbank' features. MFCC: mel frequency cepstral
        coefficients; FBANK: mel-log filterbank energies (default 'fbank')
    win_size_ms : int or float
        Window length in milliseconds for Fourier transform to be applied
        (default 20)
    win_shift_ms : int or float 
        Window overlap in milliseconds; default set at 50% window size 
        (default 10)
    num_filters : int
        Number of mel-filters to be used when applying mel-scale. For 
        'fbank' features, 20-128 are common, with 40 being very common.
        (default 40)
    num_mfcc : int
        Number of mel frequency cepstral coefficients. First coefficient
        pertains to loudness; 2-13 frequencies relevant for speech; 13-40
        for acoustic environment analysis or non-linguistic information.
        Note: it is not possible to choose only 2-13 or 13-40; if `num_mfcc`
        is set to 40, all 40 coefficients will be included.
        (default 40). 
    samplerate : int, optional
        The sample rate of the sound data or the desired sample rate of
        the wavfile to be loaded. (default None)
        
    Returns
    -------
    feats : np.ndarray
        Feature matrix, shape: (num_frames, num_filters)
    '''
    if isinstance(sound, str):
        data, sr = soundprep.loadsound(sound, samplerate=samplerate)
    else:
        if samplerate is None:
            raise ValueError('No samplerate given. Either provide filename or appropriate samplerate.')
        data, sr = sound, samplerate
    win_samples = int(win_size_ms * sr // 1000)
    if 'fbank' in features:
        feats = logfbank(data,
                         samplerate=sr,
                         winlen=win_size_ms * 0.001,
                         winstep=win_shift_ms * 0.001,
                         nfilt=num_filters,
                         nfft=win_samples)
    elif 'mfcc' in features:
        feats = mfcc(data,
                     samplerate=sr,
                     winlen=win_size_ms * 0.001,
                     winstep=win_shift_ms * 0.001,
                     nfilt=num_filters,
                     numcep=num_mfcc,
                     nfft=win_samples)
    return feats

if __name__=='__main__':

    import time


    batch_size = 30
    num_frames = 12
    start = time.time()
    clean_feats = get_feats_directory(wavefile_directory = \
        '/home/airos/Projects/Data/denoising_sample_data/clean_speech/', 
                    feature_type='fbank', win_size_ms = 16, win_shift_ms = 8, 
                    num_features=40, frames_per_sample = num_frames, batch_size=batch_size)

    end1 = time.time()
    noisy_feats = get_feats_directory(wavefile_directory = \
        '/home/airos/Projects/Data/denoising_sample_data/noisy_speech/', 
                    feature_type='fbank', win_size_ms = 16, win_shift_ms = 8, 
                    num_features=40, frames_per_sample = num_frames, batch_size=batch_size)
    end2 = time.time()

    print('Duration clean feature extraction: ', end1 - start)
    print('Duration noisy feature extraction: ', end2 - end1)
    print('Total duration: ', end2 - start)

    assert clean_feats.shape == noisy_feats.shape
    # TODO: better check for ensuring data is correctly aligned

    print(clean_feats.shape)
    print(noisy_feats.shape)

    np.save('clean_speech_batchsize{}_numframes{}.npy'.format(batch_size, num_frames), clean_feats)
    np.save('noisy_speech_batchsize{}_numframes{}.npy'.format(batch_size, num_frames), noisy_feats)


    # visualize random sample to inspect them:
    visualize_feats(clean_feats[5,:,:,:].reshape((
            clean_feats.shape[1]*clean_feats.shape[2],
            clean_feats.shape[3])),
        feature_type='fbank')

    visualize_feats(noisy_feats[5,:,:,:].reshape((
            clean_feats.shape[1]*clean_feats.shape[2],
            clean_feats.shape[3])),
        feature_type='fbank')


    #visualize_feats(noisy_feats[4],
        #feature_type='fbank')
    ##visualize_feats(clean_feats[4],
        ##feature_type='fbank')
