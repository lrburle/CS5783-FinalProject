"""
Oklahoma State University, ECE
Author(s): Landon Burleson, Madhusti Dhasaradhan, and Alex Sensintaffar

This class is used to generate and gather train, test, and possible validation data for the
Transformer based RNN model.

"""

import numpy as np
from scipy import signal, misc

class Data:
		def get_Train(self):
			x_train = [1, 2, 3]
			y_train = [1, 2, 3]
   
			return x_train, y_train

		def get_Test(self):
			x_test = [1, 2, 3]
			y_test = [1, 2, 3]
   
			return x_test, y_test
		
		def noise_add(self, x):
			return x