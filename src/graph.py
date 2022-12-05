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
    
    def lossEpochs(self, model_name):
        plt.figure(figsize=[12, 9])
        plt.title("Loss vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(self.history.history['mean_squared_error'])
        plt.grid()
        plt.savefig(f'{model_name}_lossvsepochs.png')

    def mseEpochs(self, model_name):
        plt.figure(figsize=[12, 9])
        plt.title("MSE vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.plot(self.history.history['mean_squared_error'])
        plt.grid()
        plt.savefig(f'{model_name}_msevsepochs.png')