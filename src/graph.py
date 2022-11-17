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
    
    def lossEpochs(self):
        plt.figure(figsize=[12, 9])
        plt.title("Loss vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.plot(self.history.history['mean_squared_error'])
        # plt.legend([f'= {test1[1]:.4}', f'= {test2[1]:.4}', f'Architecture - 10 Layer CNN - Filter Number Hourglass, Test Accuracy = {test3[1]:.4}'], loc='lower right')
        plt.grid()
        plt.savefig('lossvsepochs.png')

    def mseEpochs(self):
        plt.figure(figsize=[12, 9])
        plt.title("MSE vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("MSE")
        plt.plot(self.history.history['mean_squared_error'])
        # plt.legend([f'Architecture - 10 Layer CNN - Filter Number Increasing, Test Accuracy = {test1[1]:.4}', f'Architecture - 10 Layer CNN - Filter Number Decreasing, Test Accuracy = {test2[1]:.4}', f'Architecture - 10 Layer CNN - Filter Number Hourglass, Test Accuracy = {test3[1]:.4}'], loc='lower right')
        plt.grid()
        plt.savefig('msevsepochs.png')
