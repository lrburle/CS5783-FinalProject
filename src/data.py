"""
Oklahoma State University, ECE
Author(s): Landon Burleson, Madhusti Dhasaradhan, and Alex Sensintaffar

This class is used to generate and gather train, test, and possible validation data for the
Transformer based RNN model.

"""
import pathlib
import os
import librosa
import shutil
import random
import csv

import numpy as np
import tensorflow as tf
import soundfile as sf
import matplotlib.pyplot as plt

class Data:
	# def __init__(self, noisePerSound=0, soundPerWord=0, word="", installBaseData=False):
	def initilize(self, noisePerSound=10, soundPerWord=1000, numberOfWords=8):
		# if installBaseData:
		self.prepareData()
		word = ["down", "go", "left", "no", "right", "stop", "up", "yes"]

		for i in range(numberOfWords):
			self.createData(noisePerSound=noisePerSound, soundsPerWord=soundPerWord, word=word[i])	

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
	
		None

	def addNoise(self, file, word, fileIdentifier="", writeLocation=None):
		# provide file location
		signal, sr  = librosa.load(file)
		signal = librosa.util.fix_length(signal, int(sr * 1))
		RMS = (np.mean(signal**2))**.5
		noise = np.random.normal(0, RMS, signal.shape[0])
		signal_noise = signal + noise

		noiseFile = file.replace(".wav", "") + "_" + word + "_noise_" + fileIdentifier + "_.wav"
		# print(noiseFile)
		file = file.replace(".wav", "_" + word + ".wav")
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
		relativePATH = "data/mini_speech_commands/" + word + "/"

		soundsPerWordCount = 0
		for file in os.listdir(relativePATH):
			if soundsPerWordCount == soundsPerWord:
				break

			f = os.path.join(relativePATH, file)
			for i in range(noisePerSound):
				self.addNoise(f, word, str(i), "noise_data/training_data/")
			soundsPerWordCount += 1
		None

	def MoveXandYFiles(self, numOfX):
		PATH = "noise_data/training_data/"
		if not os.path.isdir(PATH + "X"):
			os.mkdir(PATH + "X/")
		if not os.path.isdir(PATH + "Y"):
			os.mkdir(PATH + "Y/")

		print("Folder X and Y created")

		fileList = []
		for f in os.listdir(PATH):
			if os.path.isfile(os.path.join(PATH, f)):
				fileList.append(f)

		for i in range(len(fileList)):
			if i % (numOfX+1) == 0:
				try:
					shutil.move(PATH +fileList[i+numOfX+1], PATH + "Y")
				except:
					None
			else:
				try:
					shutil.move(PATH + fileList[i], PATH + "X")
				except:
					None
		print("File X and Y filled")
		None

	def createTrainingTestVerificationDataSet(self):
		PATH_X = "noise_data/training_data/X/"
		PATH_Y = "noise_data/training_data/Y/"

		PATH_TRAINING_X = PATH_X + "training"
		PATH_TESTING_X = PATH_X + "testing"
		PATH_VERIFICATION_X = PATH_X + "verification"

		# Create folders
		if not os.path.isdir(PATH_TRAINING_X):
			os.mkdir(PATH_TRAINING_X)
			os.chmod(PATH_TRAINING_X, 0o777)

		if not os.path.isdir(PATH_VERIFICATION_X):
			os.mkdir(PATH_VERIFICATION_X)
			os.chmod(PATH_VERIFICATION_X, 0o777)

		if not os.path.isdir(PATH_TESTING_X):
			os.mkdir(PATH_TESTING_X)
			os.chmod(PATH_TESTING_X, 0o777)
		
		print("folders Created")

		# Transfer data into three different sub folders
		fileListX = []
		for f in os.listdir(PATH_X):
			if os.path.isfile(os.path.join(PATH_X, f)):
				fileListX.append(f)

		print(len(fileListX))

		xTrainingSize = int(len(fileListX) * .6)
		xTestingSize = int(len(fileListX) * .2)
		xVerificationSize = int(len(fileListX) * .2) + 1

		for i in range(xTrainingSize):
			index = random.randrange(0, len(fileListX)-1)
			temp = "*"
			try:
				temp = fileListX.pop(index)
				shutil.move(PATH_X + temp, PATH_TRAINING_X)
			except Exception as e:
				print("Train i", i, "temp", temp )
				print(e)

		for i in range(xTestingSize):
			index = random.randrange(0, len(fileListX)-1)
			temp = "*"
			try:
				temp = fileListX.pop(index)
				shutil.move(PATH_X + temp, PATH_TESTING_X)
			except Exception as e:
				print("Test i", i, "temp", temp)
				print(e)

		for i in range(xVerificationSize):
			temp = "*"
			try:
				index = random.randrange(0, len(fileListX)-1)
				temp = fileListX.pop(index)
				shutil.move(PATH_X + temp, PATH_VERIFICATION_X)
			# except ValueError:
			# 	shutil.move(PATH_X + fileListX.pop(0), PATH_VERIFICATION_X)
			except Exception as e:
				print("V i", i, "temp", temp)
				print(e)

		None

	def createDataListFile(self):
		PATH_X = "noise_data/training_data/X/"
		PATH_Y = "noise_data/training_data/Y/"

		PATH_TRAINING_X = PATH_X + "training"
		PATH_TESTING_X = PATH_X + "testing"
		PATH_VERIFICATION_X = PATH_X + "verification"

		xTrainingFileList = os.listdir(PATH_TRAINING_X)
		xTestingFileList = os.listdir(PATH_TESTING_X)
		xVerificationFileList = os.listdir(PATH_VERIFICATION_X)

		# string = "00b01445_nohash_0_left_noise_2_.wav"
		# print(string[0:string.index("noise")-1] + ".wav")

		with open(PATH_X + "xTrainingFileList.csv", 'w', newline='') as f:
			writer = csv.writer(f)
			for file in xTrainingFileList:
				yFile = file[0:file.index("noise")-1] + ".wav"
				writer.writerow([file, yFile])
		
		with open(PATH_X + "xTestingFileList.csv", 'w', newline='') as f:
			writer = csv.writer(f)
			for file in xTestingFileList:
				yFile = file[0:file.index("noise")-1] + ".wav"
				writer.writerow([file, yFile])

		with open(PATH_X + "xVerificationFileList.csv", 'w', newline='') as f:
			writer = csv.writer(f)
			for file in xVerificationFileList:
				yFile = file[0:file.index("noise")-1] + ".wav"
				writer.writerow([file, yFile])

	def plotSound(self, file):
		signal, sr = librosa.load(file)
		plt.plot(signal)
		plt.show()

	def createDataSets(self, noisesPerSound=10, soundPerNoise=1000, numberOfWords=8):
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


		self.initilize(noisesPerSound, soundPerNoise, numberOfWords)
		self.MoveXandYFiles(noisesPerSound)
		self.createTrainingTestVerificationDataSet()
		self.createDataListFile()


# ************************* Functions to get training, testing, and verification data 
	def get_Data(self, dataType):
		PATH_X = "noise_data/training_data/X/"
		PATH_Y = "noise_data/training_data/Y/"		
		
		PATH_DATA_X = PATH_X + dataType + "/"

		fileListX = []
		for f in os.listdir(PATH_DATA_X):
			if os.path.isfile(os.path.join(PATH_DATA_X, f)):
				fileListX.append(f)

		# randomize data
		randomFileListX = []
		for i in fileListX:
			randomFileListX.append(fileListX.pop(random.randrange(0, len(fileListX))))
		
		x_data = []
		y_data = []
		data_rate_x = []
		data_rate_Y = []

		for file in randomFileListX:
			waveform, dataRate = librosa.load(PATH_DATA_X + file)
			x_data.append(waveform)
			data_rate_x.append(dataRate)
			# print(type(waveform), waveform)

			yFile = file[0:file.index("noise")-1] + ".wav"
			waveform, dataRate = librosa.load(PATH_Y + yFile)
			y_data.append(waveform)
			data_rate_Y.append(dataRate)

		x_data = np.stack(x_data)
		y_data = np.stack(y_data)
		# y_data = np.concatenate(y_data)

		return x_data, y_data, data_rate_x, data_rate_Y

	def get_Train(self, sequence_length):
		x_data, y_data, data_rate_x, data_rate_Y = self.get_Data("training")
		x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)
		y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)

		x_data = x_data[:, :sequence_length, :]
		y_data = y_data[:, :sequence_length, :]
  
		return x_data, y_data, data_rate_x, data_rate_Y

	def get_Test(self, sequence_length):
		x_data, y_data, data_rate_x, data_rate_Y = self.get_Data("testing")

		x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)
		y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)

		x_data = x_data[:, :sequence_length, :]
		y_data = y_data[:, :sequence_length, :]

		return x_data, y_data, data_rate_x, data_rate_Y

	def get_Verification(self, sequence_length):
		x_data, y_data, data_rate_x, data_rate_Y = self.get_Data("verification")

		x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)
		y_data = y_data.reshape(y_data.shape[0], y_data.shape[1], 1)
		
		x_data = x_data[:, :sequence_length, :]
		y_data = y_data[:, :sequence_length, :]

		return x_data, y_data, data_rate_x, data_rate_Y
	
	def convertMatrixToWav(self, M, sr, model_name, data_type):
		if not os.path.exists('./outputs'):
			os.mkdir('outputs')

		for idx, row in enumerate(M):
			sf.write(f'./outputs/{model_name}_{data_type}_wav_{idx}.wav', row, sr)

	def plotOutputSounds(self):
		fileLocation = "C:/Users/alexs/Downloads/"
		actural, sr0, = librosa.load(fileLocation + "rnn_actual_wav_0.wav")
		inputs, sr1 = librosa.load(fileLocation + "rnn_inputs_wav_0.wav")
		predicted, sr2 = librosa.load(fileLocation + "rnn_predicted_wav_0.wav")

		plt.figure("Actural")
		plt.plot(actural)

		plt.figure("Input")
		plt.plot(inputs)

		plt.figure("predicted")
		plt.plot(predicted)

		plt.show()





