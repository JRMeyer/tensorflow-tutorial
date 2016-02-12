from __future__ import division
import numpy as np
import tensorflow as tf
import tarfile
import os
import matplotlib.pyplot as plt
import time



def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

def import_data():
    if "data" not in os.listdir(os.getcwd()):
        # Untar directory of data if we haven't already
        tarObject = tarfile.open("data.tar.gz")
        tarObject.extractall()
        tarObject.close()
        print("Extracted tar to current directory")
    else:
        # we've already extracted the files
        pass

    print("loading training data")
    trainX = csv_to_numpy_array("data/trainX.csv", delimiter="\t")
    trainY = csv_to_numpy_array("data/trainY.csv", delimiter="\t")
    print("loading test data")
    testX = csv_to_numpy_array("data/testX.csv", delimiter="\t")
    testY = csv_to_numpy_array("data/testY.csv", delimiter="\t")
    return trainX,trainY,testX,testY


###################
### IMPORT DATA ###
###################

trainX,trainY,testX,testY = import_data()


#########################
### GLOBAL PARAMETERS ###
#########################

# number of times we iterate through training data
numEpochs = 1000       #tensorboard shows that accuracy plateaus at ~25k epochs  (maybe even at ~19k epochs ==> +.002)
# here we set the batch size to be the total number of emails in our training
# set... if you have a ton of data you can adjust this so you don't load
# everything in at once
batchSize = trainX.shape[0]
# a smarter learning rate for gradientOptimizer
# learningRate = tf.train.exponential_decay(learning_rate=0.001,
learningRate = tf.train.exponential_decay(learning_rate=0.0008,     #TODO figure out what ideal learning rate is
                                          global_step= 1,
                                          decay_steps=trainX.shape[0],
                                          decay_rate= 0.95,
                                          staircase=True)

# Get our dimensions for our different variables and placeholders:
# numFeatures = the number of words extracted from each email
numFeatures = trainX.shape[1]
# numLabels = number of classes we are predicting (here just 2: ham or spam)
numLabels = trainY.shape[1]

#create a tensorflow session
sess = tf.Session()


####################
### PLACEHOLDERS ###
####################

# X = X-matrix / feature-matrix / data-matrix... It's a tensor to hold our email
# data. 'None' here means that we can hold any number of emails
X = tf.placeholder(tf.float32, [None, numFeatures])
# yGold = Y-matrix / label-matrix / labels... This will be our correct answers
# matrix. Every row has either [1,0] for SPAM or [0,1] for HAM. 'None' here 
# means that we can hold any number of emails
yGold = tf.placeholder(tf.float32, [None, numLabels])



#################
### VARIABLES ###
#################

#all values are randomly assigned:
    #sqrt(6 / (numInputNodes + numOutputNodes + 1))

weights = tf.Variable(tf.random_normal([numFeatures,numLabels],
                                       mean=0,
                                       stddev=(np.sqrt(6/numFeatures+
                                                         numLabels+1)),
                                       name="weights"))

bias = tf.Variable(tf.random_normal([1,numLabels],
                                    mean=0,
                                    stddev=(np.sqrt(6/numFeatures+numLabels+1)),
                                    name="bias"))



########################
### OPS / OPERATIONS ###
########################

apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias") 
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")
#summary op for regression output
activation_summary_OP = tf.histogram_summary("output", activation_OP)

# Mean squared error cost function
cost_OP = tf.nn.l2_loss(activation_OP-yGold, name="squared_error_cost")
#summary op for cost
cost_summary_OP = tf.scalar_summary("cost", cost_OP)

training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

# argmax(activation_OP, 1) gives the label our model thought was most likely
# argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(yGold,1))

# False is 0 and True is 1, what was our average?
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))
#summary op for accuracy
accuracy_summary_OP = tf.scalar_summary("accuracy", accuracy_OP)





# Initializes everything we've defined made above, but doesn't run anything
# until sess.run()
init_OP = tf.initialize_all_variables()



###########################
### GRAPH LIVE UPDATING ###
###########################

#lists to hold values for live graphing
epoch_values = []
accuracy_values = []

#TODO build two figures so that we can track decreasing cost at same time

#set up matplotlib for live updating
plt.ion()
plt.show()
plt.xlabel("number of epochs")
plt.ylabel("accuracy %")
plt.title("accuracy on training data")


#####################
### RUN THE GRAPH ###
#####################

# Initialize all tensorflow objects
sess.run(init_OP)

# summary ops to check how the variables (W, b) are updating after each iteration
# weightSummary = tf.histogram_summary("weights", weights.eval(session=sess))
weightSummary = tf.scalar_summary("weights", tf.reduce_mean(weights.eval(session=sess)))
biasSummary = tf.histogram_summary("biases", bias.eval(session=sess))

#merge all summaries
all_summary_OPS = tf.merge_all_summaries()

#summary writer
writer = tf.train.SummaryWriter("summary_logs", sess.graph_def)

#initialize reporting variables
cost = 0
diff = 1

#training epochs
for i in range(numEpochs):
    if i > 1 and diff < .0001:
        print("change in cost %g; convergence."%diff)
        break
    else:
        #run training step
        step = sess.run(training_OP, feed_dict={X: trainX, yGold: trainY})
        #report occasional stats
        if i % 10 == 0:
            #add epoch to epoch_values
            epoch_values.append(i)
            #generate accuracy stats on test data
            summary_results, train_accuracy, newCost = sess.run([all_summary_OPS, accuracy_OP, cost_OP], feed_dict={X: trainX, yGold: trainY})
            #add accuracy to live graphing variable
            accuracy_values.append(train_accuracy)
            # accuracy_values = accuracy_values + ([train_accuracy] * 9)
            #write summary stats to writer
            writer.add_summary(summary_results, i)
            #re-assign values for variables
            diff = abs(newCost - cost)
            cost = newCost
            #generate print statements
            print("step %d, training accuracy %g"%(i, train_accuracy))
            print("step %d, cost %g"%(i, newCost))
            print("step %d, change in cost %g"%(i, diff))
            plt.plot(epoch_values, accuracy_values)
            plt.draw()
            time.sleep(1)


# How well did we do overall?
print("final accuracy on test set: %s" %str(sess.run(accuracy_OP, feed_dict={X: testX, yGold: testY})))


##############################
### SAVE TRAINED VARIABLES ###
##############################

#create Saver
saver = tf.train.Saver()
#save variables to .ckpt file
# saver.save(sess, "trained_variables.ckpt")


############################
### MAKE NEW PREDICTIONS ###
############################

#close tensorflow session
sess.close()

#to view tensorboard:
    #1. run: tensorboard --logdir=/path/to/log-directory
    #2. open your browser to http://localhost:6006/
#see tutorial here for graph visualization https://www.tensorflow.org/versions/0.6.0/how_tos/graph_viz/index.html
