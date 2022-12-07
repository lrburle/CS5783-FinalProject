"""
Oklahoma State University, ECE
Author(s): Landon Burleson, Madhusti Dhasaradhan, and Alex Sensintaffar

This code is used to output graphs for user visualization.
"""
import matplotlib.pyplot as plt
import numpy as np

class Graph:
    def __init__(self, history):
        self.history = history
    
    def mseEpochs(self, model_name, test):
        plt.figure(figsize=[12, 9])
        plt.title(f'MSE vs Epochs - ({model_name.upper()}) - Test Accuracy - {test[1]}', fontsize=18)
        plt.xlabel("Epochs", fontsize=15)
        plt.ylabel("MSE", fontsize=15)

        plt.plot(self.history.history['mean_squared_error'], 'b', label='Mean Squared Error')

        plt.grid()
        plt.savefig(f'{model_name}_msevsepochs.png')