import numpy as np
import tensorflow as tf
import tarfile
import os

#import data
def untar(tarPath):
    tarObject = tarfile.open(tarPath)
    tarObject.extractall()
    tarObject.close()
    print "Extracted tar to current directory"

def importData(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

#untar data file
if "data" in os.listdir(os.getcwd()):
    untar("data.tar.gz")

#imports data with following dimensions
print("loading training data")
trainX = importData("data/trainX.csv", delimiter="\t")
trainY = importData("data/trainY.csv", delimiter="\t")
print("loading test data")
testX = importData("data/testX.csv", delimiter="\t")
textY = importData("data/testY.csv", delimiter="\t")

#set global parameters
epochs = 5                                                              #number of times we iterate through training data
batchSize = len(trainX)                                                 #batchSize of n
learningRate = tf.train.exponential_decay(                              #used by gradientOptimizer
                                            learning_rate=0.01,
                                            global_step= 1,
                                            decay_steps=len(trainX),
                                            decay_rate= 0.95,
                                            staircase=True)


#set tensorflow placeholders
# input = tf.placeholder(dtype="float", name="Input", shape=[None, ])
# label = tf.placeholder(dtype="float", name="Label", shape=[None, ])