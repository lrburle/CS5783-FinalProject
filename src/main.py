"""
Oklahoma State University, ECE
Author(s): Landon Burleson, Madhusti Dhasaradhan, and Alex Sensintaffar

This is the main code needed to import data, train/test model, and evaluate.
"""

from model import Model
from data import Data

if __name__ == '__main__':
	   
    # Import the data needed
    
    
    # Create, train, test, and evaluate model
    m = Model()
    model = m.model()
    m.train(model)
    m.save(model)
    m.predict

    # Output necessary graphs and outputs. 
    
