# Python Script for converting the CSV Data back to PNG Segments

from __future__ import absolute_import, division, print_function, unicode_literals

import csv
import xlsxwriter
import imageio
from PIL import ImageFilter
from PIL import ImageEnhance
from PIL import Image
import numpy as np
import librosa.display
import pylab
import librosa
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Don't display the picture

# Declare full paths of .csv files to read/write from/to. (DEPRECIATED)
# Need to make it so that it goes through ALL csv files
preprocessedData = "/Users/Smiles/Documents/GitHub/TF_CNN/Datasets/PrePro_Data/1.csv"

# CSV File Path
# This will change based on which computer is running the program
dataFolder = "/Users/Smiles/Documents/GitHub/TF_CNN/Datasets/PrePro_Data/*.csv"

# Partitions pixels into segements for individual record storage


def divideChunks(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]

# Reconstruct the image to ensure it's correct


def buildImage(state):
    checkData = csv.reader(
        open(preprocessedData, 'r', encoding="utf8"), delimiter=",")
    i = 0
    for record in checkData:
        list1 = []
        tensor = []
        for pixel in range(2, 3842):
            list1.append(record[pixel])
        # print(len(list1))   #TEMP
        eachColumn = list(divideChunks(list1, 8))
        array = np.array(eachColumn, dtype=np.uint8)
        print(array)  # TEMP
        print(array.shape)  # TEMP
        tensor.append(array)
        print(np.asarray(tensor).shape)
        newImage = Image.fromarray(array)
        if state == 1:
            # If the "State" vriable is set to 1, the system will output the generated images
            newImage.show()
        i += 1
    # return newImage
    return np.asarray(tensor)

# Read Label Information from CSV - [ID, Genre]


def readLabel(fileNumber):
    reader = csv.reader(open(fileNumber, 'r', encoding="utf8"), delimiter=",")
    i = 0
    for row in reader:
        label = [row[0], row[1]]
        i += 1
        if i == 2:
            return label  # This should prevent the loop from running 80 times

# Collect ALL Track Labels into an Array - [[ID 1, Genre 1], [ID 2, Genre 2], ...]


def getLabels():
    labels = []
    for fileNum in sorted(glob.glob(dataFolder), key=lambda name: int(name[(len(dataFolder) - 5):-4])):
        labels.append(readLabel(fileNum))
    return labels

# TESTING PORTION***

#allLabels = getLabels()
# for i in range(0, len(allLabels)):  #Prints ALL Genres ONLY**
    # print(allLabels[i][1])


#X_data = buildImage(0)
#print('X_data Shape: ', np.array(X_data).shape)
# print(X_data)
