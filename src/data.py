"""
Oklahoma State University, ECE
Author(s): Landon Burleson, Madhusti Dhasaradhan, and Alex Sensintaffar

This class is used to generate and gather train, test, and possible validation data for the
Transformer based RNN model.

"""
import pathlib
import os
import librosa

import numpy as np
import tensorflow as tf
import soundfile as sf

from scipy import signal, misc

class Data:
	def __init__(self, path, outFile):
		self.createFileList(path, outFile)
		self.prepareData()
		# Feed files
    
	def createFileList(self, path, outFile):
		dir_list = os.listdir(path)

		w = open(outFile, "w")

		for i in range(len(dir_list)):
			w.write(dir_list[i] + "\n")

	def prepareData(self):
		DATASET_PATH = 'data/mini_speech_commands'

		data_dir = pathlib.Path(DATASET_PATH)
		if not data_dir.exists():
			tf.keras.utils.get_file(
				'mini_speech_commands.zip',
				origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
				extract=True,
				cache_dir='.', cache_subdir='data')
	
		self.createFileList("data/mini_speech_commands/down/", "data/FileListDown.txt")
		self.createFileList("data/mini_speech_commands/go/", "data/FileListGo.txt")
		self.createFileList("data/mini_speech_commands/left/", "data/FileListLeft.txt")
		self.createFileList("data/mini_speech_commands/no/", "data/FileListNo.txt")
		self.createFileList("data/mini_speech_commands/right/", "data/FileListRight.txt")
		self.createFileList("data/mini_speech_commands/stop/", "data/FileListStop.txt")
		self.createFileList("data/mini_speech_commands/up/", "data/FileListUp.txt")
		self.createFileList("data/mini_speech_commands/yes/", "data/FileListYes.txt")

	def addNoise(self, file):
		# provide file location
		signal, sr  = librosa.load(file)
		RMS = (np.mean(signal**2))**.5
		noise = np.random.normal(0, RMS, signal.shape[0])
		signal_noise = signal + noise

		noiseFile = file.replace(".wav", "") + "_noise.wav"

		sf.write(noiseFile, signal_noise, sr)

	def createData(self, path):
		# TODO inter mixing word?
		# TODO how many noisy versions of the same sound
		# TODO how many sounds from same word

		None

	def get_Train(self):

		return x_train, y_train

	def get_Test(self):

		return x_test, y_test
	
	def testPrepareData():
		data = Data()
		data.prepareData()

	def testAddNoise():
		data = Data()

		file = "test_down.wav"
		data.addNoise(file)


