import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
# from CreateData import *

def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        waveform, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    # Convert the frequencies to log scale and transpose, so that the time is
    # represented on the x-axis (columns).
    # Add an epsilon to avoid taking a log of zero.
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels

if __name__ == '__main__':
    # DATASET_PATH = "data/mini_speech_commands/down/0a9f9af7_nohash_0.wav"
    DATASET_PATH = "data/mini_speech_commands"

    data_dir = pathlib.Path(DATASET_PATH)

    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=data_dir,
        batch_size=64,
        validation_split=0.2,
        seed=0,
        output_sequence_length=16000,
        subset='both')

    label_names = np.array(train_ds.class_names)

    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    sets = 6
    for example_audio, example_labels in train_ds.take(sets):  
        print(example_audio.shape)
        print(example_labels.shape)

    figure, axes = plt.subplots(2, sets,figsize=(12, 8))
    # sound = 0
    for i in range(sets):
        label = label_names[example_labels[i]]
        waveform = example_audio[i]
        spectrogram = get_spectrogram(waveform)

        timescale = np.arange(waveform.shape[0])
        axes[0][i].plot(timescale, waveform.numpy())
        axes[0][i].set_title('Waveform ' + label)
        axes[0][i].set_xlim([0, 16000])

        plot_spectrogram(spectrogram.numpy(), axes[1][i])
        axes[1][i].set_title('Spectrogram')
        plt.suptitle(label.title())
    plt.show()