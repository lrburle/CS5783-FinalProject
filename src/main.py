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


if __name__ == '__main__':
    
    # Uncomment if a new set of data is needed. 
    dat = Data()

    x_train, y_train, x_data_rate, y_data_rate = dat.get_Train()  
    x_test, y_test, x_data_rate_test, y_data_rate_test = dat.get_Test()  
    x_valid, y_valid, x_data_rate_val, y_data_rate_val = dat.get_Verification()  

    # dat.generateAndSaveData()

    # Import the data needed
    # xdata = np.loadtxt('xdata.csv', delimiter=',', dtype=float)
    # ydata = np.loadtxt('ydata.csv', delimiter=',', dtype=float)

    # # Split the data into training and testing datasets
    # x_train = xdata[0:40, :]
    # y_train = ydata[0:40, :]

    # x_test = xdata[41:-1, :]
    # y_test = ydata[41:-1, :] 

    # Create, train, test, and evaluate model
    epochs = 25
    m = Model(x_train, y_train, x_test, y_test, x_valid, y_valid, epochs)
    model = m.model()
    
    #This loads the latest model iteration
    #model = m.model_load('backup/trial.10.bak')
    
    #This loads the latest checkpoint into the model.
    #model = m.checkpoint_load('backup', model)

    history, model = m.train(model)
    m.model_save(model)

    output = m.predict(model, x_test)

    # Output necessary graphs and outputs. 
    g = Graph(history)
    g.mseEpochs()
    g.lossEpochs()
