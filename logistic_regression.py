import numpy as np
import tensorflow as tf
import tarfile
import os
import math

#import data
def untar(tarPath):
    tarObject = tarfile.open(tarPath)
    tarObject.extractall()
    tarObject.close()
    print("Extracted tar to current directory")

def importData(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

#untar data file
if "data" not in os.listdir(os.getcwd()):
    untar("data.tar.gz")

#imports data with following dimensions
print("loading training data")
trainX = importData("data/trainX.csv", delimiter="\t")
trainY = importData("data/trainY.csv", delimiter="\t")
print("loading test data")
testX = importData("data/testX.csv", delimiter="\t")
textY = importData("data/testY.csv", delimiter="\t")

#set global parameters
#number of times we iterate through training data
epochs = 5
#batchSize of n
batchSize = trainX.shape[0]
#used by gradientOptimizer
learningRate = tf.train.exponential_decay(learning_rate=0.01,
                                          global_step= 1,
                                          decay_steps=trainX.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)

#determining required dimensions
numFeatures = trainX.shape[1]
numLabels = trainY.shape[1]

#set tensorflow placeholders
    #PUT EXPLANATION HERE
# Define our tensor to hold email data. None indicates
# we can hold any number of emails
x = tf.placeholder(tf.float32, [None, numFeatures])
# This will be our correct answers/output
yGold = tf.placeholder(tf.float32, [None, numLabels])


# Create the Variables for the weights and biases
weights = tf.Variable(tf.random_normal(
                                        [numFeatures, numLabels],
                                        mean=0,
                                        stddev=math.sqrt(float(6) / float(numFeatures + numLabels + 1))),
                                        name="weights")

biases = tf.Variable(tf.random_normal(
                                        [1, numLabels],
                                        mean=0,
                                        stddev=math.sqrt(float(6) / float(numFeatures + numLabels + 1))),
                                        name="biases")

# This is our prediction output
#TODO here
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Calculate cross entropy by taking the negative sum of our correct values
# multiplied by the log of our predictions
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# This is training the NN with backpropagation
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# Initializes all the variables we made above
init = tf.initialize_all_variables()


