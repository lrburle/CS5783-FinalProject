# CS5783-FinalProject - Final project for Fall 2022 CS5783 at OSU

This project showcases the use of RNN and transformer based models to de-noise audio input samples.

## Library requirements
In order to effectively execute the code, the library 'librosa' must be installed for the audio manipulation functions needed to create the 
test, train, and validation training sets for use in the model. This library can be installed with the following command:
```bash
pip3 install librosa
```

Other libraries used: numpy, tensorflow, shutil, soundfile, matplotlib, os, csv, and random. These libraries can be installed utilizing the same command structure as above. 

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
