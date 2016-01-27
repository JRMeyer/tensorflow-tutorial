import numpy as np
import tensorflow as tf

#import data
def importData(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)


#imports data with following dimensions
trainX = importData("/data/trainX", delimiter="\t")
trainY = importData("/data/trainY", delimiter="\t")
testX = importData("/data/testX", delimiter="\t")
textY = importData("/data/testY", delimiter="\t")

#set global parameters
epochs = 5                                          #number of times we iterate through training data
batchSize = len(trainX)                             #batchSize of n
learningRate = tf.train.exponential_decay(          #used by gradientOptimizer
                                            learning_rate=0.01,
                                            global_step= 1,
                                            decay_steps=len(trainX),
                                            decay_rate= 0.95,
                                            staircase=True)