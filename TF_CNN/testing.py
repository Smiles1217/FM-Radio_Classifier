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

# load a single file as a numpy array


def load_file(filepath):
    dataframe = read_csv(filepath, header=None, delim_whitespace=True)
    return dataframe.values

# load a list of files and return as a 3d numpy array


def load_group(filenames, prefix='/Users/Smiles/Documents/GitHub/TF_CNN/Datasets/'):
    loaded = list()
    for name in filenames:
        data = load_file(prefix + name)
        loaded.append(data)
    # stack group so that features are the 3rd dimension
    loaded = np.dstack(loaded)
    return loaded

# load a dataset group, such as train or test


def load_dataset_group(group, prefix='/Users/Smiles/Documents/GitHub/TF_CNN/Datasets/'):
    filepath = prefix + group + '/Inertial Signals/'
    # load all 9 files as a single array
    filenames = list()
    # total acceleration
    filenames += ['total_acc_x_'+group+'.txt', 'total_acc_y_' +
                  group+'.txt', 'total_acc_z_'+group+'.txt']
    # body acceleration
    filenames += ['body_acc_x_'+group+'.txt', 'body_acc_y_' +
                  group+'.txt', 'body_acc_z_'+group+'.txt']
    # body gyroscope
    filenames += ['body_gyro_x_'+group+'.txt', 'body_gyro_y_' +
                  group+'.txt', 'body_gyro_z_'+group+'.txt']
    # load input data
    X = load_group(filenames, filepath)
    # load class output
    y = load_file(prefix + group + '/y_'+group+'.txt')
    print(y)
    return X, y

# load the dataset, returns train and test X and y elements


def load_dataset(prefix='/Users/Smiles/Documents/GitHub/TF_CNN/Datasets/'):
    # load all train
    trainX, trainy = load_dataset_group('train', prefix + 'HARDataset/')
    print(trainX.shape, trainy.shape)
    # load all test
    testX, testy = load_dataset_group('test', prefix + 'HARDataset/')
    print(testX.shape, testy.shape)
    # zero-offset class values
    trainy = trainy - 1
    testy = testy - 1
    # one hot encode y
    trainy = tf.keras.utils.to_categorical(trainy)
    testy = tf.keras.utils.to_categorical(testy)
    print(trainX.shape, trainy.shape, testX.shape, testy.shape)
    return trainX, trainy, testX, testy

# fit and evaluate a model


def evaluate_model(trainX, trainY, testX, testY):
    numTimesteps = trainX.shape[1]
    numFeatures = trainX.shape[2]
    numOutputs = trainY.shape[1]
    print(numOutputs)

    verbose = 1
    batchSize = 32
    epochs = 10
    strideLen = 1
    numFilters = 32
    kernSize = 3  # Width of the input image
    poolSize = 2
    #numClasses = 8
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
    model.fit(trainX, trainY, epochs=epochs,
              batch_size=batchSize, validation_data=(testX, testY), verbose=verbose)


trainX, trainY, testX, testY = load_dataset()
evaluate_model(trainX, trainY, testX, testY)

dtypeXtr = type(trainX)
dtypeYtr = type(trainY)
dtypeXte = type(testX)
dtypeYte = type(testY)

print(dtypeXtr, dtypeYtr, dtypeXte, dtypeYte)

# print(trainX)
# print(trainY)
print(testX)
print(testY)
print(trainX.shape, trainY.shape, testX.shape, testY.shape)
