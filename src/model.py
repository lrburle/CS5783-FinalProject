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
	def __init__(self, x_train, y_train, x_test, y_test,x_valid, y_valid, epochs):
		self.x_train = x_train
		self.y_train = y_train
		self.x_test = x_test
		self.y_test = y_test
		self.x_valid = x_valid
		self.y_valid = y_valid
		self.epochs = epochs

	# Defining our model.
	def model(self):
		out = tf.keras.Sequential([
			keras.layers.SimpleRNN(20, return_sequences=True, input_shape=(None, 1)),
			keras.layers.SimpleRNN(20, return_sequences=True),
			# keras.layers.Attention(),

			#Dense layer output.
			keras.layers.Dense(30, activation='relu'),
			keras.layers.Dense(1, activation='linear')
			])
		
		opt = keras.optimizers.Adam(learning_rate=0.001)

		# out.compile(loss="mse", optimizer=opt, metrics=['sparse_categorical_accuracy'])
		out.compile(loss="mse", optimizer=opt, metrics=['mean_squared_error', 'accuracy'])
		self.model = out

		return out 
	
	# Training our model
	def train(self, modelIn):
		checkpoint_path = './backup/cp.1.ckpt'

		while(os.path.exists(checkpoint_path)):
			incr = checkpoint_path.split(".")[-2]
			next_incr = int(incr) + 1
			checkpoint_path = checkpoint_path.replace(incr, str(next_incr))

		cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='mean_squared_error', mode=min, save_weights_only=True)
  
		history = modelIn.fit(self.x_train, self.y_train, epochs=self.epochs, callbacks=[cp_callback], validation_data=(self.x_valid, self.y_valid))

		return history, modelIn

	#Saving progress on the model to come back to previous iterations. 
	def model_save(self, modelIn):
		path = './backup/trial.1.bak'
		while(os.path.exists(path)):
			incr = path.split(".")[-2]
			next_incr = int(incr) + 1
			path = path.replace(incr, str(next_incr))

		modelIn.save(path)
	
	def model_load(self, path):
		return tf.keras.models.load_model(path)
	
	def checkpoint_load(self, dir, modelIn):
		return modelIn.load_weights(tf.train.latest_checkpoint(dir))
	
	def model_evaluate(self, modelIn):
		evaluate = modelIn.evaluate(self.x_test, self.y_test)
		return evaluate 

	def predict(self, model, x_test):
		#Code for outputting the RNN sequence from the input. Hopefully filtered.
		predict = model.predict(x_test)
		return predict
