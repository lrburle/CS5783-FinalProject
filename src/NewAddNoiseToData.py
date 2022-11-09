import librosa
import os
import matplotlib.pyplot as plt
import numpy as np
import math

def createFileList(path, outFile):
    dir_list = os.listdir(path)

    w = open(outFile, "w")

    for i in range(len(dir_list)):
        w.write(dir_list[i] + "\n")

def prepareFiles():
    createFileList("A:/OSU/Semester 9/CS 5783/CS5783-FinalProject/src/data/mini_speech_commands/down/", "data/FileListDown.txt")
    createFileList("A:/OSU/Semester 9/CS 5783/CS5783-FinalProject/src/data/mini_speech_commands/go/", "data/FileListGo.txt")
    createFileList("A:/OSU/Semester 9/CS 5783/CS5783-FinalProject/src/data/mini_speech_commands/left/", "data/FileListLeft.txt")
    createFileList("A:/OSU/Semester 9/CS 5783/CS5783-FinalProject/src/data/mini_speech_commands/no/", "data/FileListNo.txt")
    createFileList("A:/OSU/Semester 9/CS 5783/CS5783-FinalProject/src/data/mini_speech_commands/right/", "data/FileListRight.txt")

def main():
    word = "down"
    locationWord = word[0].upper() + word[1:len(word)]

    f = open("data/FileList" + locationWord + ".txt", 'r')
    lines = [line.rstrip() for line in f][1]

    location = "A:/OSU/Semester 9/CS 5783/CS5783-FinalProject/src/data/mini_speech_commands/" + word + "/" + lines
    # print(lines)
    signal, sr = librosa.load(location)
    
    # plt.plot(signal)
    fig, axes = plt.subplots(3, 1)
    axes[0].plot(signal)
    axes[0].set_title(lines)

    RMS = math.sqrt(np.mean(signal**2))
    # RMS_n = math.sqrt(RMS**2/(10**(SNR/10)))
    noise=np.random.normal(0, RMS, signal.shape[0])
    axes[1].plot(noise)
    axes[1].set_title("noise")

    signal_noise = signal+noise
    axes[2].plot(signal_noise)
    axes[2].set_title("signal noise")

    plt.show()

if __name__ == '__main__':
    main()

# prepareFiles()