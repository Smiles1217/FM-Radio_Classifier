# Version 1 of a TensorFlow CNN

from __future__ import absolute_import, division, print_function, unicode_literals

import CSV_to_PNG
import tensorflow as tf
from PIL import Image
import csv
import xlsxwriter
import imageio
from PIL import ImageFilter
from PIL import ImageEnhance
import numpy as np
import librosa.display
import pylab
import librosa
import os
import glob
import matplotlib
import matplotlib.pyplot as plt
from pandas import read_csv
import keras_preprocessing
matplotlib.use('Agg')  # Don't display the picture


def load_file(filepath):
    dataframe = read_csv(filepath, header=None)
    # print(dataframe)  # TEMP
    # print(dataframe.to_numpy())  # TEMP
    return dataframe.to_numpy()


def load_group(filenames, prefix='/Users/Smiles/Documents/GitHub/TF_CNN/Datasets/PrePro_Data/'):
    values = list()
    labels = list()
    i = 0  # TEMP - will need to make cleaner in the future
    print(filenames)  # TEMP
    for name in filenames:
        data = load_file(prefix + name)
        while i < 80:
            values.append(data[i][2:])
            labels.append(2)  # TEMP - Forcing Label
            # labels.append(data[i][1])
            i += 1
    values = np.dstack(values)
    values = np.transpose(values, (2, 1, 0))
    labels = np.dstack(labels)
    labels = np.squeeze(labels, axis=0)
    return values, labels


def load_dataset_group(group, prefix='/Users/Smiles/Documents/GitHub/TF_CNN/Datasets/PrePro_Data/'):
    filepath = prefix + group + '/'
    filenames = list()
    # Need to implement a loop to go through all CSV files
    filenames += ['1.csv']
    # print(filenames)  # TEMP
    X, Y = load_group(filenames, filepath)
    return X, Y


def load_dataset(prefix='/Users/Smiles/Documents/GitHub/TF_CNN/Datasets/PrePro_Data/'):
    trainX, trainY = load_dataset_group('train', prefix)
    #print(trainX.shape, trainY.shape)
    testX, testY = load_dataset_group('test', prefix)
    #print(testX.shape, testY.shape)
    # trainY = trainY - 1
    # testY = testY - 1
    trainY = tf.keras.utils.to_categorical(trainY)
    testY = tf.keras.utils.to_categorical(testY)
    trainY = np.squeeze(trainY, axis=0)
    testY = np.squeeze(testY, axis=0)
    #print(trainX.shape, trainY.shape, testX.shape, testY.shape)
    return np.asarray(trainX), np.asarray(trainY), np.asarray(testX), np.asarray(testY)


def evaluate_model(trainX, trainY, testX, testY):
    numTimesteps = trainX.shape[1]
    numFeatures = trainX.shape[2]
    numOutputs = trainY.shape[1]

    verbose = 1
    batchSize = 1
    epochs = 10
    strideLen = 1
    numFilters = 32
    kernSize = 3  # Width of the input image
    poolSize = 2
    # numClasses = 8
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
    #print(type(trainX), type(trainY), type(testX), type(testY))
    model.fit(trainX, trainY, epochs=epochs,
              batch_size=batchSize, validation_data=(testX, testY), verbose=verbose)  # *****


# classes = ['Electronic', 'Experimental', 'Folk',
# 'Hip-Hop', 'Instrumental', 'Pop', 'Rock']

trainX, trainY, testX, testY = load_dataset()

#dtypeXtr = type(trainX)
#dtypeYtr = type(trainY)
#dtypeXte = type(testX)
#dtypeYte = type(testY)

#print(dtypeXtr, dtypeYtr, dtypeXte, dtypeYte)

trainX = tf.convert_to_tensor(trainX, dtype=tf.int8)
trainY = tf.convert_to_tensor(trainY, dtype=tf.int8)
testX = tf.convert_to_tensor(testX, dtype=tf.int8)
testY = tf.convert_to_tensor(testY, dtype=tf.int8)

#dtypeXtr = type(trainX)
#dtypeYtr = type(trainY)
#dtypeXte = type(testX)
#dtypeYte = type(testY)

#print(dtypeXtr, dtypeYtr, dtypeXte, dtypeYte)

evaluate_model(trainX, trainY, testX, testY)

# print(trainX)
# print(trainY)
# print(testX)
# print(testY)
#print(trainX.shape, trainY.shape, testX.shape, testY.shape)
