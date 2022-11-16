"""
Oklahoma State University, ECE
Author(s): Landon Burleson, Madhusti Dhasaradhan, and Alex Sensintaffar

This is the main code needed to import data, train/test model, and evaluate.
"""
from model import Model 
from data import Data
from graph import Graph 

import numpy as np
import os
from scipy.io import wavfile

def generateAndSaveData():
    xfiles = os.listdir('./training_data_subset/training_data/X/')
    yfiles = os.listdir('./training_data_subset/training_data/Y/')

    dat = Data()
    xdata, ydata = dat.convertToArray('./training_data_subset/training_data', xfiles, yfiles)

    np.savetxt('xdata.csv', xdata, delimiter=',')
    np.savetxt('ydata.csv', ydata, delimiter=',')

if __name__ == '__main__':
    
    # Uncomment if a new set of data is needed. 
    # generateAndSaveData()

    # Import the data needed
    xdata = np.loadtxt('xdata.csv', delimiter=',', dtype=float)
    ydata = np.loadtxt('ydata.csv', delimiter=',', dtype=float)

    x_train = xdata[0:40, :]
    y_train = ydata[0:40, :]

    x_test = xdata[41:-1, :]
    y_test = ydata[41:-1, :] 

    # Create, train, test, and evaluate model
    epochs = 25
    m = Model(x_train, y_train, x_test, y_test, epochs)
    model = m.model()
    history, model = m.train(model)
    m.model_save(model)
    output = m.predict(model, x_test)

    # Output necessary graphs and outputs. 
    g = Graph(history)
    g.mseEpochs()
    g.lossEpochs()