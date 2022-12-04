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

if __name__ == '__main__':
    load_model_flag = False 
    train_flag = False
    model_directory = './backup'

	# Accepts command line arguments for controlling the 
    if (len(sys.argv) > 1):
        load_model_flag, model_directory, train_flag, model_type = cmdlParse(sys.argv)
   
    dat = Data()

    x_train, y_train, x_data_rate, y_data_rate = dat.get_Train()  
    x_test, y_test, x_data_rate_test, y_data_rate_test = dat.get_Test()
    x_valid, y_valid, x_data_rate_val, y_data_rate_val = dat.get_Verification()

    # Create, train, test, and evaluate model
    epochs = 25
    m = Model(x_train, y_train, x_test, y_test, x_valid, y_valid, epochs)

    if model_type == 'tran':
        model = m.buildTransformer(vector_in=x_train, h_size=256, num_h=4, num_of_blocks=4, dense_units=[128], dropout=0.25, dense_dropout=0.4)
    else:
        model = m.model()

    if (load_model_flag):
        # This loads the latest model iteration
        model = m.model_load(model_directory)
    
    #This loads the latest checkpoint into the model.
    #model = m.checkpoint_load('backup', model)

    if (train_flag):
        history, model = m.train(model)
        m.model_save(model)
    
    if (train_flag or load_model_flag):
        output = m.predict(model, x_test)
        dat.convertMatrixToWav(output, x_data_rate_test[0])

    # Output necessary graphs and outputs. 
    g = Graph(history)
    g.mseEpochs()
    g.lossEpochs()