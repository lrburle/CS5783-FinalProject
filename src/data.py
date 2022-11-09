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

	def __init__(self, noisePerSound, soundsPerWord, word, installBaseData=False):
		if installBaseData:
			self.prepareData()

		self.createData(noisePerSound, soundsPerWord, word)

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
	
		# self.createFileList("data/mini_speech_commands/down/", "data/FileListDown.txt")
		# self.createFileList("data/mini_speech_commands/go/", "data/FileListGo.txt")
		# self.createFileList("data/mini_speech_commands/left/", "data/FileListLeft.txt")
		# self.createFileList("data/mini_speech_commands/no/", "data/FileListNo.txt")
		# self.createFileList("data/mini_speech_commands/right/", "data/FileListRight.txt")
		# self.createFileList("data/mini_speech_commands/stop/", "data/FileListStop.txt")
		# self.createFileList("data/mini_speech_commands/up/", "data/FileListUp.txt")
		# self.createFileList("data/mini_speech_commands/yes/", "data/FileListYes.txt")
		None

	def addNoise(self, file, word, fileIdentifier="", writeLocation=None):
		# provide file location
		signal, sr  = librosa.load(file)
		RMS = (np.mean(signal**2))**.5
		noise = np.random.normal(0, RMS, signal.shape[0])
		signal_noise = signal + noise

		noiseFile = file.replace(".wav", "") + "_noise_" + fileIdentifier + "_.wav"
		# print(noiseFile)

		if writeLocation != None:
			slashIndex = writeLocation.index('/')
			if not os.path.isdir(writeLocation[0:slashIndex]):
				os.mkdir(writeLocation[0:slashIndex])
			if not os.path.isdir(writeLocation):
				os.mkdir(writeLocation)
			if not os.path.isdir(writeLocation + word):
				os.mkdir(writeLocation + word)

			file = file.replace("data/mini_speech_commands/", writeLocation)
			noiseFile = noiseFile.replace("data/mini_speech_commands/", writeLocation)
			if not os.path.isfile(file):
				sf.write(file, signal, sr)

		sf.write(noiseFile, signal_noise, sr)

	def createData(self, noisePerSound, soundsPerWord, word):
		# TODO inter mixing word?
		# TODO how many noisy versions of the same sound
		# TODO how many sounds from same word

		relativePATH = "data/mini_speech_commands/" + word + "/"

		soundsPerWordCount = 0
		for file in os.listdir(relativePATH):
			if soundsPerWordCount > soundsPerWord:
				break

			f = os.path.join(relativePATH, file)
			for i in range(noisePerSound):
				self.addNoise(f, word, str(i), "noise_data/training_data/")
			soundsPerWordCount += 1
		None

	def get_Train(self):
		x_train = [1, 2, 3]
		y_train = [1, 2, 3]

		return x_train, y_train

	def get_Test(self):
		x_test = [1, 2, 3]
		y_test = [1, 2, 3]

		return x_test, y_test
	
	def noise_add(self, x):
		return x

def testPrepareData():
	data = Data()
	data.prepareData()

def testAddNoise():
	data = Data()

	file = "test_down.wav"
	data.addNoise(file)

def testCreateData():
	data = Data()

	data.createData(noisePerSound=10, soundPerWord=10, word="down")

# Example for creating testing and training data with noise
def testInitilization():
	# Set installBaseData to True if tensorflow data has not been installed 
	Data(noisePerSound=10, soundPerNoise=10, word="down", installBaseData=False)

