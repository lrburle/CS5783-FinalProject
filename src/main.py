"""
Oklahoma State University, ECE
Author(s): Landon Burleson, Madhusti Dhasaradhan, and Alex Sensintaffar

This is the main code needed to import data, train/test model, and evaluate.

Command line option flags:
    -t : is used to train the model
    -m tran : model selection. Ommitting this option will use the default RNN model. 'tran' in
    combination with -m will select the transformer model.
    -l : loads a selected model from the ./backup/ directory.

To use the command line arguments, you may do so using the following command formats:
    python3 main.py -t -m tran // This is used to train the transformer model.
    python3 main.py -l ./backup/trail.4.bak -m tran // This is used to load a former model for the transformer model. This assumes the traial.4.bak directory is associated with the transformer model. 

"""
from model import Model 
from data import Data
from graph import Graph 

import numpy as np
import sys

# Used to control the top level from the terminal.
def cmdlParse(args):
    load_model_flag = False 
    train_flag = False
    model_directory = None
    model_type = 'null'

    for i in range(len(args)):
        if '-m' in args[i]:
            model_type = args[i+1]
        if '-l' in args[i]:
            load_model_flag = True
            model_directory = args[i+1]
        if '-t' in args[i]:
            train_flag = True
    
    return load_model_flag, model_directory, train_flag, model_type

# The main function of code for the repository. 
if __name__ == '__main__':
    LOAD_MODEL_FLAG = False 
    TRAIN_FLAG = False
    MODEL_DIRECTORY = './backup'

	# Accepts command line arguments for controlling the 
    if (len(sys.argv) > 1):
        LOAD_MODEL_FLAG, MODEL_DIRECTORY, TRAIN_FLAG, MODEL_TYPE = cmdlParse(sys.argv)
   
    dat = Data()

    sequence_length = 22048

	# Sets up the training, test, and validation data for use in the training process.
    x_train, y_train, x_data_rate, y_data_rate = dat.get_Train(sequence_length)  
    x_test, y_test, x_data_rate_test, y_data_rate_test = dat.get_Test(sequence_length)
    x_valid, y_valid, x_data_rate_val, y_data_rate_val = dat.get_Verification(sequence_length)

    # Create, train, test, and evaluate model
    epochs = 25
    # Creates the Model object for use in training the RNN or Transformer Model.
    m = Model(x_train, y_train, x_test, y_test, x_valid, y_valid, epochs)

	# Based on the command line arguments will determine if the desired model is the transformer or
	# RNN model. Please refer to the README commands in the repository. 
    if MODEL_TYPE == 'tran':
        model = m.transformer_model(h_size=256, num_h=4, num_of_blocks=4, dense_units=[128], dropout=0.25, dense_dropout=0.4)
    else:
        MODEL_TYPE = 'rnn'
        model = m.rnn_model()

	# This will load the model based upon the directory given to the command line argument -l
    if (LOAD_MODEL_FLAG):
        # This loads the latest model iteration
        model = m.model_load(MODEL_DIRECTORY)
        test = model.evaluate(x_test, y_test)

        print(f'Evaluation metrics: \nAccuracy : {test[1]}\nLoss : {test[0]}')        
    
    # Trains the chosen model if the -t flat is set upon execution of the main.py script.
    if (TRAIN_FLAG):
        history, model = m.train(model)
        test = model.evaluate(x_test, y_test)

        print(f'Evaluation metrics: \nAccuracy : {test[1]}\nLoss : {test[0]}')        
        m.model_save(model, MODEL_TYPE)

        # Output necessary graphs and outputs. 
        g = Graph(history)
        g.mseEpochs(MODEL_TYPE, test)
    
    if (TRAIN_FLAG or LOAD_MODEL_FLAG):
        output = m.predict(model, x_test)
        dat.convertMatrixToWav(x_test, x_data_rate_test[0], MODEL_TYPE, data_type='inputs')
        dat.convertMatrixToWav(output, x_data_rate_test[0], MODEL_TYPE, data_type='predicted')
        dat.convertMatrixToWav(y_test, y_data_rate_test[0], MODEL_TYPE, data_type='actual')