from __future__ import print_function
import librosa
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') # Don't display the picture
import pylab
import librosa.display
import numpy as np
from PIL import Image 
from PIL import ImageEnhance
from PIL import ImageFilter
import imageio
import xlsxwriter
import csv

#DIVIDE CHUNKS FUNCTION 
#Partitions pixels into segements for individual record storage
def divideChunks(list, n):
    for i in range(0, len(list), n):
        yield list[i:i + n]
        
        
        
#MAIN FUNCTION
#Declare full paths of .csv files to read/write from/to.
preprocessedData = "C:/Users/Bonzi/Desktop/ECE487_MusicClassifier-master/ECE487_MusicClassifier-master/Preprocessed Data Files/"
metaData = "C:/Users/Bonzi/Desktop/Music-Classification-master/fma/tracks.csv"
q = 0

#Get an array of songs in order from each folder in fma small folder.
folders = os.listdir("C:/Users/Bonzi/Desktop/fma_small/fma_small")
for folder in folders:
    #Gets the files from each folder.
    files = os.listdir("C:/Users/Bonzi/Desktop/fma_small/fma_small/" + folder)
    for file in files: 
        
        #GET THE MATCHING GENRE AND TRACK ID: 
        #Remove leading 0s from each file to match track id in tracks.csv
        #and remove the .mp3 extention to get the full track id.
        trackID = file.lstrip("0")
        trackID = trackID.rstrip(".mp3")      
        #Open the metaData file tracks.csv in read mode.
        metaDataFile = csv.reader(open(metaData, 'rt', encoding = "utf8"), delimiter=",")
        #Loop though each record in metaData file until the record with the specific track ID is found
        for record in metaDataFile:
            #If the track id for the .mp3 file matches the track id for the record
            if trackID == record[0]:
                #Get the genere for the track.
                genre = record[40]
        
        #GET PIXEL DATA
        #Get full path for each .mp3 file
        fullPath = "C:/Users/Bonzi/Desktop/fma_small/fma_small/" + folder + "/" + file
        #Create a .CSV file for each song.
        preprocessedData = "C:/Users/Bonzi/Desktop/ECE487_MusicClassifier-master/ECE487_MusicClassifier-master/Preprocessed Data Files/" + trackID + ".csv"
        #Load the audio file into an array. Return array and sample rate.
        audioArray, sampleRate = librosa.load(fullPath, sr = 30000)
        #Create a name for the picture.
        savePath = 'C:/Users/Bonzi/Desktop/Pythonspectrogram.png'
        #Remomve unnessisary attribute of the plot to isolate the spectrogram only.
        pylab.axis('off') 
        pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) 
        spectrogram = librosa.feature.melspectrogram(audioArray, sampleRate)
        librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max))
        pylab.savefig(savePath, bbox_inches=None, pad_inches=0)
        pylab.close()
        #Open the image and convert it to greyscale
        img = Image.open(savePath).convert('L')
        #Rotate image so that so that the time is on the y axis. 
        #The purpose of this is that it makes getting each 8 columns of pixels easier. (Now rows)
        rotatedImage = img.rotate(-90, expand = 1)
        #Get an array of pixel data
        data = list(rotatedImage.getdata())
        #Initalize number of pixels per record
        numPixelsPerRecord = 3840
        #Send list to function to be partitioned into pixel portions of records. 
        pixelDataPerRecord = list(divideChunks(data, numPixelsPerRecord))
        #Reformat pixels so that they're easier to work with in the CNN
        for pixelList in pixelDataPerRecord:
            eachColumn = list(divideChunks(pixelList, 480))
            array = np.array(eachColumn, dtype=np.uint8)
            newImage = Image.fromarray(array)
            newImage = newImage.rotate(90, expand = 1)
            finalDataFormat = list(newImage.getdata())
            
            #GENERATE AND SAVE EACH RECORD ENTRY
            recordEntry = [trackID, genre]
            for pixel in finalDataFormat:
                recordEntry.append(pixel)
            #Append the record entry into the preprocessed data file.
            with open(preprocessedData, 'a', newline = "") as fd:
                writer = csv.writer(fd)
                writer.writerow(recordEntry)
        q = q + 1
        if q == 6:
            quit()
            
            
            
            
            
            
            
            
            
            
            
        #NOTE: THIS PORTION OF THE ALGORITHM CAN BE USED TO GET THE PIXEL DATA IN THE 
        #CORRECT FORMAT FOR THE CNN!    
                
        # #Reconstruct the image to ensure it's correct
        # checkData = csv.reader(open(preprocessedData, 'r', encoding = "utf8"), delimiter = ",")
        # for record in checkData:
        #     list1 = []
        #     for pixel in range (2, 3842):
        #         list1.append(record[pixel])
        #     print(len(list1))
        #     eachColumn = list(divideChunks(list1, 8))
        #     array = np.array(eachColumn, dtype=np.uint8)
        #     newImage = Image.fromarray(array).show()


            



