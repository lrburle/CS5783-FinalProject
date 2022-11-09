"""
Oklahoma State University, ECE
Author(s): Landon Burleson, Madhusti Dhasaradhan, and Alex Sensintaffar

This python file is used to create a RNN used within the background noise removal process for
use in audio tracks/streams

The model class shown below is used to build and utilize the built architecture in this python file. 
"""

import numpy as np
import tensorflow as tf
import os
from tensorflow import keras

class Model:
	def __init__(self, x_train, y_train, x_test, y_test, x_valid, y_valid, epochs):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.x_valid = x_valid
		self.y_valid = y_valid
		self.epochs = epochs

	def model(self):
		out = tf.keras.Sequential([
			keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
			keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]),
			keras.layers.TimeDistributed(keras.layers.Dense(10))
			])
		
		opt = keras.optimizers.Adam(learning_rate=0.001)

		out.compile(loss="mse", optimizer=opt, metrics=["accuracy"])

		return out 
	
	def train(self, modelIn):
		checkpoint_path = './backup/cp.1.ckpt'

		while(os.path.exist(checkpoint_path)):
			incr = checkpoint_path.split(".")[-2]
			next_incr = int(incr) + 1
			checkpoint_path = checkpoint_path.replace(incr, str(next_incr))

		cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
  
		history = modelIn.fit(self.x_train, self.y_train, epochs=self.epochs, callbacks=[cp_callback])

		return history

	def model_save(self, modelIn, incr):
		path = './backup/trial.1.bak'
		while(os.path.exist(path)):
			incr = path.split(".")[-2]
			next_incr = int(incr) + 1
			path = path.replace(incr, str(next_incr))

		modelIn.save(path)

	def predict(self):
		#Code for outputting the RNN sequence from the input. Hopefully filtered.
		return 
