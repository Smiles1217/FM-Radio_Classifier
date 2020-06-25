# -----------------------------------------------------------------------------
# Version 1.2 of a TensorFlow CNN
# -----------------------------------------------------------------------------

from __future__ import absolute_import, division, print_function, unicode_literals

#import CSV_to_PNG
import tensorflow as tf
from PIL import Image
import csv
#import xlsxwriter
import imageio
from PIL import ImageFilter
from PIL import ImageEnhance
import numpy as np
import librosa.display
import pylab
import librosa
import os
import sys
import glob
import time
import matplotlib
import matplotlib.pyplot as plt
from pandas import read_csv
import keras_preprocessing
matplotlib.use('Agg')  # Don't display the picture

# TEMP - Print the entire NumPy Array for Testing
# np.set_printoptions(threshold=sys.maxsize)

# Temporary implementation of a single Path definition:
# **I plan to change this to utilize a "local" path call for increased generatlity**
PATH = 'C:/Users/bbons/OneDrive/Desktop/Music Classifer current/Music Classifier/PrePro_Data/'

# Global Network Variables for easy Universal Adjustment:
tr_batchSize = 1    # Batch Size variable for training data - TEMP
ts_batchSize = 1    # Batch Size variable for testing/evaluation data - TEMP
pr_batchSize = 1    # Batch Size variable for prediction data - TEMP
tota_batches = 159
epochs = 10
verbose = 1

strideLen = 1
numFilters = 32
kernSize = 3  # Width of the input image
poolSize = 2
# numClasses = 8    # Not sure if necessary

# ---------------------------------------------------------------------------------------------------------
# Function for converting the labels that are strings into an integer representation based on a given key:


# def label_to_int(genre, prefix=PATH):
#     labels = load_file(prefix + 'genre_labels.txt', True)
#     i = 0

#     while(i < 8):
#         if labels[i][1] == genre.upper():
#             # If the genre is foumd, return with the subsequent key and break the loop early
#             return labels[i][0]
#         i += 1

#     # If the genre is not found in the key, return (-1) to indicate an error
#     return -1

# Genre Labels (for reference):
#1 - ELECTRONIC
#2 - EXPERIMENTAL
#3 - FOLK
#4 - HIP-HOP
#5 - INSTRUMENTAL
#6 - INTERNATIONAL
#7 - POP
#8 - ROCK


# -----------------------------------------------------------------------------
# Function for loading CSV data from a file:


# def load_file(filepath, D_whitespaceTF):
#     dataframe = read_csv(filepath, header=None,
#                          delim_whitespace=D_whitespaceTF)
#     return dataframe.to_numpy()

# -----------------------------------------------------------------------------
# Function for literally loading the value and label data:


def group_data(filepath):
    values = list()
    labels = list()
    #directory = os.fsencode(prefix)
    # This will need to be determined based on how many rows per file we decide to read.
    #i = 0
    reader = csv.reader(open(filepath, "r"))
    for row in reader:
    #while i < tr_batchSize:
        # In case files outside of the desired data accidentally end up in the directory
        #data = load_file(filepath, False)
        values.append(row[2:])
        labels.append(row[1])
        # i += 1
        # if i > 1939
        #     break
    values = np.dstack(values)
    values = np.transpose(values, (2, 1, 0))
    labels = np.dstack(labels)
    labels = np.squeeze(labels, axis=0)
    return values, labels

# -----------------------------------------------------------------------------
# Function for loading the values and labels groups as X and Y respectively:


def load_dataset_group(group, batch_iter, prefix=PATH):
    filepath = prefix + group + '/batch' + str(batch_iter) + '.csv'
    X, Y = group_data(filepath)
    return X, Y

# -----------------------------------------------------------------------------
# Function to load an entire dataset:


def load_dataset(batch_iter, prefix=PATH):
    trainX, trainY = load_dataset_group('train_1', batch_iter, prefix)
    testX, testY = load_dataset_group('test_1', 80, prefix)
    # trainY = trainY - 1   # Unsure if necessary
    # testY = testY - 1
    trainY = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)
    trainY = np.squeeze(trainY, axis=0)
    testY = np.squeeze(testY, axis=0)
    return np.asarray(trainX), np.asarray(trainY), np.asarray(testX), np.asarray(testY)

# --------------------------------------------------------------------------------
# Function for building a CNN model:


def create_model(trainX, trainY, testX, testY):
    numTimesteps = trainX.shape[1]
    numFeatures = trainX.shape[2]
    numOutputs = trainY.shape[1]

    inputShape = (numTimesteps, numFeatures)
    activFunc = 'relu'

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(numFilters, kernSize, strides=strideLen,
                               activation=activFunc, input_shape=inputShape),
        tf.keras.layers.MaxPool1D(pool_size=poolSize),
        tf.keras.layers.Conv1D(2*numFilters, kernSize, strides=strideLen,
                               activation=activFunc),
        tf.keras.layers.MaxPool1D(pool_size=poolSize),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation=activFunc),
        tf.keras.layers.Dense(numOutputs, activation='softmax')
    ])
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model

# --------------------------------------------------------------------------------
# Function for Training a CNN model:


def train_model(model, trX, trY, tsX, tsY):
    # Trains a constructed model based on the given training and validation data.
    model.fit(trX, trY, epochs=epochs,
              batch_size=tr_batchSize, validation_data=(tsX, tsY), verbose=verbose)
    #model.fit(trX, trY, epochs=epochs,
    #          batch_size=tr_batchSize, verbose=verbose)             

# --------------------------------------------------------------------------------
# Function for Testing a CNN model:


def test_model(model, trX, trY, tsX, tsY):
    # Tests an already trained model based on an evaluation dataset specified by
    # the passed in parameters tsX and tsY (Data and Labels) respectively.

    # **batch_size specification is probably unecessary since we would be evaluating
    #   with a different dataset to training and thus possibly a differnt batch size**
    model.evaluate(tsX, tsY, epochs=epochs,
                   batch_size=ts_batchSize, verbose=verbose)


# --------------------------------------------------------------------------------
# Function for Predicting Classification for a single input on a CNN model:
# **This is the function that will be used for real-time radio data testing!**


def predict_model(model, inputData):

    # Input for the predict function needs to be a list, hence the square brackets
    # Input is also expected to have been prepared to meet expected CNN input

    # **This may need to be changed to actually use or operate similarly to the
    #   load_dataset() method to ensure the input shape and type are correct**

    # **batch_size specification is probably unecessary since we would be predicting
    #   based on one input from the radio reciever at a time**
    model.predict([inputData], batch_size=pr_batchSize, verbose=verbose)

# --------------------------------------------------------------------------------
# Function for Saving a CNN model:


def save_model(model, model_name, batch_iter):
    # Defines a string with the current date and time
    #curTime = time.strftime("%Y%m%d-%H%M%S")
    modelName = str(model_name) + "_CNN.model"
    print(modelName)

    # Saves the model in the same directory as this code
    model.save(modelName)
    print("Model: " + modelName + " saved")

    #Update number batches we got though in .txt file.
    writer = csv.writer(open(PATH + "batch_iter.txt", 'a', newline = ""))
    writer.writerow(str(batch_iter))

# --------------------------------------------------------------------------------
# Function for Loading a CNN model:


def load_model(model_name):
    # Loads a model based on the given name
    # (extension must also be passed as part of the string)
    model = tf.keras.models.load_model(model_name)
    return model


# -----------------------------------------------------------------------------
# Testing Portion (main):
# -----------------------------------------------------------------------------

#Define iterator
i = 121
j = 1
# Load the dataset:
# X is the Pixel Values
# Y is the Labels
trainX, trainY, testX, testY = load_dataset(i)

# Convert data to tensors - **Should make this it's own function**
trainX = tf.convert_to_tensor(trainX, dtype=tf.int8)
trainY = tf.convert_to_tensor(trainY, dtype=tf.int8)
testX = tf.convert_to_tensor(testX, dtype=tf.int8)
testY = tf.convert_to_tensor(testY, dtype=tf.int8)

# Build the Model:
CovNet1 = create_model(trainX, trainY, testX, testY)

# Loop though batches
while i <= tota_batches:
    # Save the Model:
    CovNet1 = load_model(str(j) + "_CNN.model")
    print("Start training")
    train_model(CovNet1, trainX, trainY, testX, testY)
    print("Finished training")
    save_model(CovNet1, j, i)
    i += 1
    j += 1
    if(j > 3):
        j = 1
    trainX, trainY, testX, testY = load_dataset(i)
    trainX = tf.convert_to_tensor(trainX, dtype=tf.int8)
    trainY = tf.convert_to_tensor(trainY, dtype=tf.int8)
    testX = tf.convert_to_tensor(testX, dtype=tf.int8)
    testY = tf.convert_to_tensor(testY, dtype=tf.int8)

# Load a Model (**Commented out for code testing**):
#CovNet1 = load_model("20200310-111535_CNN.model")

# Train a Model - TEMP (**Commented out for code testing**):
#train_model(CovNet1, trainX, trainY, testX, testY)

# Test a Model - TEMP (**Commented out for code testing**):
#test_model(CovNet1, trainX, trainY, testX, testY)

# Predict with a Model - TEMP (**Commented out for code testing**):
# predict_model(CovNet1, "/Users/Smiles/Documents/GitHub/TF_CNN/Datasets/PrePro_Data/test_1/1.csv")

# Printing out the actual data and data shape for testing: - TEMP
# print(trainX)
# print(trainY)
# print(testX)
# print(testY)
# print(trainX.shape, trainY.shape, testX.shape, testY.shape)
