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

    # Defining our RNN model.
    def rnn_model(self):
        out = tf.keras.Sequential([
            keras.layers.SimpleRNN(64, return_sequences=True, input_shape=(self.x_train.shape[1], 1)),
            keras.layers.Dropout(0.4),

            #Dense layer output.
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(1)
        ])
        
        opt = keras.optimizers.Adam(learning_rate=0.001)

        # out.compile(loss="mse", optimizer=opt, metrics=['sparse_categorical_accuracy'])
        out.compile(loss="mse", optimizer=opt, metrics=['mean_squared_error', 'accuracy'])
        out.summary()

        self.model = out
  
        return out 
    
    def encoderLayer(self, inputs, numHeads, dropout, units):
        # MultiHead Attention layer.
        att = keras.layers.MultiHeadAttention(num_heads=numHeads, key_dim=self.x_train[2])(inputs)

        do1 = keras.layers.Dropout(dropout=dropout)(att)
        x = keras.layers.LayerNormalization(epsilon=1e-6)(inputs + do1)

        # Feedforward network
        d = keras.layers.Dense(units=units, activation='relu')(x)
        d = keras.layers.Dense(units=1)(d) #Not specifiying the type of activation function defaults to a linear function.
        d = keras.layers.Dropout(dropout=dropout)(d)

        output = keras.layers.LayerNormalization(epsilon=1e-6)(x + d) 

        return output 

    def transformer_model(self, h_size, num_h, num_of_blocks, dense_units, dropout, dense_dropout):
        # input_shape = [self.x_train.shape[1], 1] #Number of samples, Number of features.
        # inputs = keras.Input(shape=input_shape)

        inputs = tf.keras.Input(shape=(None, self.x_train.shape[1]), name='inputs')

        x = inputs

        x = keras.layers.Dense(self.x_train.shape[2], use_bias=True, activation='linear')(x)

        x *= tf.math.sqrt(tf.cast(self.x_train.shape[2], tf.float32))
        x = keras.layers.PositionEmbedding(self.x_train[1], self.x_train[2])(x)
        x = keras.layers.Dropout(dropout=dropout)(x)
  
        for i in range(num_of_blocks):
            x = self.encoderLayer(inputs=x, num_heads=num_h, headSize=h_size, dropout=dropout, units=dense_units)

        outputs = keras.layers.Dense(1)(x)

        model = keras.Model(inputs, outputs)

        opt = keras.optimizers.Adam(learning_rate=0.00005)
        model.compile(loss="mse", optimizer=opt, metrics=['mean_squared_error', 'accuracy'])

        model.summary()

        self.model = model

        return model

    # Training our model
    def train(self, modelIn):
        checkpoint_path = './backup/cp.1.ckpt'

        while(os.path.exists(checkpoint_path)):
            incr = checkpoint_path.split(".")[-2]
            next_incr = int(incr) + 1
            checkpoint_path = checkpoint_path.replace(incr, str(next_incr))

        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor='mean_squared_error', mode=min, save_weights_only=True)
  
        history = modelIn.fit(self.x_train, self.y_train, epochs=self.epochs, callbacks=[cp_callback], validation_data=(self.x_valid, self.y_valid), batch_size=16)

        return history, modelIn

    #Saving progress on the model to come back to previous iterations. 
    def model_save(self, modelIn, name):
        path = f'./backup/{name}.trial.1.bak'
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