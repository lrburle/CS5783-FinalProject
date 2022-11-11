"""
Oklahoma State University, ECE
Author(s): Landon Burleson, Madhusti Dhasaradhan, and Alex Sensintaffar

This class is used to generate and gather train, test, and possible validation data for the
Transformer based RNN model.

"""
import pathlib
import os
import librosa
import pydub

import numpy as np
import tensorflow as tf
import soundfile as sf
import matplotlib.pyplot as plt

from scipy import signal, misc

class Data:

	# def __init__(self, noisePerSound=0, soundPerWord=0, word="", installBaseData=False):
	def initilize(self, noisePerSound, soundPerWord=0, word="", installBaseData=False):

		if installBaseData:
			self.prepareData()

		self.createData(noisePerSound, soundPerWord, word)

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

	def addNoise(self, file, fileIdentifier="", writeLocation=None):
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
			# if not os.path.isdir(writeLocation + word):
			# 	os.mkdir(writeLocation + word)

			file = file.replace("data/mini_speech_commands/", writeLocation)
			noiseFile = noiseFile.replace("data/mini_speech_commands/", writeLocation)
			
			file = file.replace("/down", "")
			file = file.replace("/up", "")
			file = file.replace("/left", "")
			file = file.replace("/stop", "")
			file = file.replace("/right", "")
			file = file.replace("/yes", "")
			file = file.replace("/no", "")
			file = file.replace("/go", "")	

			noiseFile = noiseFile.replace("/down", "")
			noiseFile = noiseFile.replace("/up", "")
			noiseFile = noiseFile.replace("/left", "")
			noiseFile = noiseFile.replace("/stop", "")
			noiseFile = noiseFile.replace("/right", "")
			noiseFile = noiseFile.replace("/yes", "")
			noiseFile = noiseFile.replace("/no", "")
			noiseFile = noiseFile.replace("/go", "")

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
				self.addNoise(f, str(i), "noise_data/training_data/")
			soundsPerWordCount += 1
		None

	def convertWavToMp3(self, file):
		sound = pydub.AudioSegment.from_wav(file)
		file = file.replace(".wav", ".mp3")
		tempFile = file.replace("training_data", "mp3_training_data")
		print("*********** tempFile", tempFile)
		file = file.replace("noise_data/training_data/", "")
		sound.export(tempFile, format="mp3")
		# return sound

	def get_Train(self):
		# both 1D
		# mix words
		# 16,000 points for testing
		# 16,000 points for verification
		# 48,000 points for training
		# 10 noise per sound
		# save matrix as .np array in noise_data (not training data) (np.save)
		# save as string in relatvie path
		# truncate the start of the data to all be same length
		# Cut it down under 100MB
		# np.matrix is full of .wav or .mp3 (which ever is smaller)

		allDataX = np.matrix([None]*80000)
		allDataY = np.matrix([None]*80000)
		PATH = "noise_data/training_data/"

		fileList = os.listdir(PATH)
		tempY = fileList[10]
		indexCount = 0

		for i in range(len(fileList)):
			if i % 11 == 1:
				tempY = PATH + "/" + fileList[i+10]
			else:
				allDataX[indexCount] = fileList[i]
				allDataY[indexCount] = tempY
				indexCount += 1
			


		# x_train = [1, 2, 3]
		# y_train = [1, 2, 3]

		return x_train, y_train

	def get_Test(self):
		x_test = [1, 2, 3]
		y_test = [1, 2, 3]

		return x_test, y_test
	
	def noise_add(self, x):
		return x

	def plotSound(self, file):
		signal, sr = librosa.load(file)
		plt.plot(signal)
		# plt.show()

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
	data = Data()

	noisePerSound = 10
	soundPerWord = 1000
	word = ["down", "go", "left", "no", "right", "stop", "up", "yes"]

	for i in word:
		data.createData(noisePerSound=noisePerSound, soundsPerWord=soundPerWord, word=i)
	
def testConvertWavToMp3():
	data = Data()

	# for file in os.listdir("noise_data/training_data"):
	fileList = os.listdir("noise_data/training_data")

	# for i in range(len(fileList)):
	for i in range(2):
		temp = "noise_data/training_data/" + fileList[i]
		print("temp", temp)
		data.convertWavToMp3(temp)

def testPlotSound():
	data = Data()
	plt.figure("Sound")
	data.plotSound("A:/OSU/Semester 9/CS 5783/CS5783-FinalProject/src/noise_data/training_data/0a9f9af7_nohash_0.wav")
	plt.figure("Noise Sound")
	data.plotSound("A:/OSU/Semester 9/CS 5783/CS5783-FinalProject/src/noise_data/training_data/0a9f9af7_nohash_0_noise_9_.wav")
	plt.show()


