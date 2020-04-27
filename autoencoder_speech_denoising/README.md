## Simple Speech Denoising Autoencoder 

In the code I refer to models of type 1 and 2. I used <a href="https://www.machinecurve.com/index.php/2019/12/19/creating-a-signal-noise-removal-autoencoder-with-keras/#creating-the-autoencoder">this</a> post to build **model type 1** (as well as plotting the reconstructed signals) and <a href="https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f">this</a> post to build **model type 2**. So far, model type 1 takes longer to train but *seems* to output a better output. This is with limited training and no audio output to compare. Neither of these posts apply the models to speech data.

The current version cleans the signal using mel filterbank energy features, similar to this <a href="https://www.isca-speech.org/archive/archive_papers/interspeech_2013/i13_0436.pdf">paper</a>. No audio signal is produced yet: only pictures. Further developments are in the works. 

Currently, Keras is used to train and implement the models. 

### Installation

```
$ python3 -m venv env
$ source env/bin/activate
(env).. $ pip install -r requirements.txt
```

### Data Extraction

There are sample data available for convenience, i.e. to easily get the model training on your machine.

Code is available to create your own datasets. Eventually I will offer instructions here.

### Train Autoencoder

First decide which model to train, 1 or 2. Default is 1. Change this in the script train_autoencoder.py, currently around line 55.

```
(env)..$ python3 train_autoencoder.py
```

With the small sample data, this should only take around 30 seconds. Note, the training stops if validation loss does not decrease after 5 epochs. 

The best model is saved in the models directory, model log in the model_logs directory.

### Load Trained Model - Reconstruct training data signals

In the script load_autoencoder_reconstructions.py put in the name of the model you want to load, around line 18. Default model is a model trained on the sample dataset.

```
(env)..$ python3 load_autoencoder_reconstructions.py
```

This will save reconstructed images of cleaned noisy signals from the test dataset in the folder images. Currently the images show mel filterbank energy (fbank) features. Default number of reconstructions is 2.

## Example Images

The speech and noise samples used are from the CD available with the book: C Loizou, P. (2013). Speech Enhancement: Theory and Practice.

### Clean speech signal:

![Imgur](https://i.imgur.com/C3jjBEk.png)

### Noisy speech signal (car background noise)

![Imgur](https://i.imgur.com/rXog9eq.png)

### Autoencoder type 1 reconstruction

Autoencoder was trained on appx. 400 of the speech recordings.

![Imgur](https://i.imgur.com/tvHbIVo.png)

### Autoencoder type 2 reconstruction

Autoencoder was trained on appx. 400 of the speech recordings.

![Imgur](https://i.imgur.com/mweqrfO.png)
