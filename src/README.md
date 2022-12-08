# Source Code
This directory houses all code needed to accomplish the proposed project. Please see the below descriptions for the python files and necessary outside modules.  

    1. main.py
    2. data.py
    3. model.py
    4. graph.py
    
The __main.py__ script is used to run all operations. Please refer to the __Running__ section shown below for in depth instructions on using the command line flags for various operations. 

The __data.py__ script is used to generate the desired training, testing, and validations sets. These are automatically set up to reproduce all experimental results shown in the project report(s).

The __model.py__ script is used to define the RNN and transformer models as well as the functions used to train, evaluate, and save models in an appropriate manner. 

The __graph.py__ script was used to output the MSE values versus the number of epoch iterations for training.

## Running
Use the following command line formats to produce the desired results from the scripts described in the previous section. 

Traininng the RNN-based model:
```bash
python3 main.py -t 
```
Training the transformer based model:
```bash
python3 main.py -t -m tran
```
Loading a previous model for the RNN model:
```bash
python3 main.py -l ./backup/rnn.trial.1.bak
```
Loading a previous model for the transformer based model:
```bash
python3 main.py -l ./backup/tran.trial.1.bak -m tran
```
__Note, due to time constraints, the transformer based model is present in the model.py code but it is not functional. The command line flags are functional and do work as intended but the model does not function appropriately.__

## Contributors
    1. Landon Burleson, Oklahoma State University ECE Department
    2. Madhusti Dhasaradhan, Oklahoma State University ECE Department
    3. Alex Sensintaffar, Oklahoma State University ECE Department
