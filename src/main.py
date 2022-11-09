"""
Oklahoma State University, ECE
Author(s): Landon Burleson, Madhusti Dhasaradhan, and Alex Sensintaffar

This is the main code needed to import data, train/test model, and evaluate.
"""
from model import Model
from data import Data

import numpy as np
from scipy.io import wavfile

if __name__ == '__main__':
    # Import the data needed
    datarate, x_train = wavfile.read('./test_down_noise.wav')
    datarate, y_train = wavfile.read('test_down.wav')

    x_train = np.matrix(x_train).T
    y_train = x_train
    # y_train = np.matrix(y_train)

    x_test = x_train
    y_test = y_train

    # Create, train, test, and evaluate model
    epochs = 10
    m = Model(x_train, y_train, x_test, y_test, epochs)
    model = m.model()
    m.train(model)
    m.save(model)
    m.predict(x_test)

    # Output necessary graphs and outputs. 
    
