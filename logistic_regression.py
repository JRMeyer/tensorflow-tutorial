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
numFeatures = trainX.shape[1]
numLabels = trainY.shape[1]
#number of times we iterate through training data
epochs = 5
#batchSize of n
batchSize = len(trainX)
#used by gradientOptimizer
learningRate = tf.train.exponential_decay(learning_rate=0.01,
                                          global_step= 1,
                                          decay_steps=len(trainX),
                                          decay_rate= 0.95,
                                          staircase=True)

# Define our tensor to hold email data. 16384 = pixels, None indicates
# we can hold any number of emails
x = tf.placeholder(tf.float32, [None, numFeatures])
# Create the Variables for the weights and biases
W = tf.Variable(tf.zeros([numFeatures, numLabels]))
b = tf.Variable(tf.zeros([numLabels]))
# This is our prediction output
y = tf.nn.softmax(tf.matmul(x, W) + b)
# This will be our correct answers/output
y_ = tf.placeholder(tf.float32, [None, numLabels])
# Calculate cross entropy by taking the negative sum of our correct values
# multiplied by the log of our predictions
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# This is training the NN with backpropagation
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# Initializes all the variables we made above
init = tf.initialize_all_variables()


#set tensorflow placeholders
input = tf.placeholder(dtype="float", name="Input", shape=[None, ])
label = tf.placeholder(dtype="float", name="Label", shape=[None, ])
