# CS5783-FinalProject - Final project for Fall 2022 CS5783 at OSU

This project showcases the use of RNN and transformer based models to de-noise audio input samples.

## Library requirements
In order to effectively execute the code, the library 'librosa' must be installed for the audio manipulation functions needed to create the 
test, train, and validation training sets for use in the model. This library can be installed with the following command:
```bash
pip3 install librosa
```

Other libraries used: numpy, tensorflow, shutil, soundfile, matplotlib, os, csv, and random. These libraries can be installed utilizing the same command structure as shown below. 
```bash
pip3 install numpy
pip3 install tensorflow
pip3 install shutil
pip3 install soundfile
pip3 install matplotlib
```
## Running the python scripts from the __src__ directory

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
The output test data is stored in the __outputs__ subdirectory inside of the __src__ directory. The input audio track is denoted *inputs*, predicted values are denoted *predicted*, and the clean audio track is denoted as *actual*. The output MSE vs Epochs graph is stored inside of the __src__ directory and named __rnn_msevsepochs.png__ for easy viewing.

__Note, due to time constraints, the transformer based model is present in the model.py code but it is not functional. The command line flags are functional and do work as intended but the model does not function appropriately.__

## Contributors
    1. Landon Burleson, Oklahoma State University ECE Department
    2. Madhusti Dhasaradhan, Oklahoma State University ECE Department
    3. Alex Sensintaffar, Oklahoma State University ECE Department
