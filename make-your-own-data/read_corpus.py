# Joshua Meyer (2017)
# <jrmeyer.github.io>

# USAGE: $ python3 ./read_corpus.py -ham data/ham -spam data/spam

# Script expects data in the format:
# Where the dirnames 'ham' and 'spam'
# Are imporant. Filenames are not.
#
#  data/
#     ham/
#       hamText1.txt
#       hamText2.txt
#       hamText3.txt
#
#     spam/
#       spamText1.txt
#       spamText2.txt
#       spamText3.txt


import glob
import random
import re
from collections import Counter
import numpy as np

def create_bag_of_words(filePaths):
    '''
    Input:
      filePaths: Array. A list of absolute filepaths
    Returns:
      bagOfWords: Array. All tokens in files
    '''
    bagOfWords = []
    # this regex filtering is specific to original dataset
    regex = re.compile("X-Spam.*\n")
    for filePath in filePaths:
        with open(filePath, encoding ="latin-1") as f:
            raw = f.read()
            raw = re.sub(regex,'',raw)
            tokens = raw.split()
            for token in tokens:
                bagOfWords.append(token)
    return bagOfWords

def get_feature_matrix(filePaths, featureDict):
    '''
    create feature/x matrix from multiple text files
    rows = files, cols = features
    '''
    featureMatrix = np.zeros(shape=(len(filePaths),
                                      len(featureDict)),
                               dtype=float)
    regex = re.compile("X-Spam.*\n")
    for i,filePath in enumerate(filePaths):
        with open(filePath, encoding ="latin-1") as f:
            _raw = f.read()
            raw = re.sub(regex,'',_raw)
            tokens = raw.split()
            fileUniDist = Counter(tokens)
            for key,value in fileUniDist.items():
                if key in featureDict:
                    featureMatrix[i,featureDict[key]] = value
    return featureMatrix

def regularize_vectors(featureMatrix):
    '''
    Input:
      featureMatrix: matrix, where docs are rows and features are columns
    Returns:
      featureMatrix: matrix, updated by dividing each feature value by the total
      number of features for a given document
    '''
    for doc in range(featureMatrix.shape[0]):
        totalWords = np.sum(featureMatrix[doc,:],axis=0)
        # some reason getting docs with 0 total words, just hacking right now
        # with +1 in denominator
        featureMatrix[doc,:] = np.multiply(featureMatrix[doc,:],(1/(totalWords+1)))

    return featureMatrix

def input_data(hamDir,spamDir,percentTest):
    ''' 
    Input:
      hamDir: String. dir of ham text files
      spamDir: String. dir of spam text file
      percentTest: Float. percentage of all data to be assigned to testset
    Returns:
      trainPaths: Array. Absolute paths to training emails
      trainY: Array. Training labels, 0 or 1 int.
      testPaths: Array. Absolute paths to testing emails
      testY: Array. Testing labels, 0 or 1 int.
    '''
    pathLabelPairs={}
    for hamPath in glob.glob(hamDir+'/*'):
        pathLabelPairs.update({hamPath:"0.,1."})
    for spamPath in glob.glob(spamDir+'/*'):
        pathLabelPairs.update({spamPath:"1.,0."})
    
    # get test set as random subsample of all data
    numTest = int(percentTest * len(pathLabelPairs))
    testing = set(random.sample(pathLabelPairs.items(),numTest))

    # delete testing data from superset of all data
    for entry in testing:
        del pathLabelPairs[entry[0]]
    
    # split training tuples of (path,label) into separate lists
    trainPaths=[]
    trainY=[]
    for item in pathLabelPairs.items():
        trainPaths.append(item[0])
        trainY.append([float(i) for i in item[1].split(',')])
    del pathLabelPairs
    trainY=np.asarray(trainY)

    # split testing tuples of (path,label) into separate lists
    testPaths=[]
    testY=[]
    for item in testing:
        testPaths.append(item[0])
        testY.append([float(i) for i in item[1].split(',')])
    del testing
    testY=np.asarray(testY)
    
    # create feature dictionary of n-grams
    bagOfWords = create_bag_of_words(trainPaths)

    # throw out low freq words if you want (set k=FreqCutOff)
    k=5
    freqDist = Counter(bagOfWords)
    newBagOfWords=[]
    for word,freq in freqDist.items():
        if freq > k:
            newBagOfWords.append(word)
    features = set(newBagOfWords)
    featureDict = {feature:i for i,feature in enumerate(features)}

    # make feature matrices
    trainX = get_feature_matrix(trainPaths,featureDict)
    testX = get_feature_matrix(testPaths,featureDict)
    
    # regularize length
    trainX = regularize_vectors(trainX)
    testX = regularize_vectors(testX)

    return trainX, trainY, testX, testY


def parse_user_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ham','--hamDir')
    parser.add_argument('-spam','--spamDir')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import sys, argparse
    # get user input
    args = parse_user_args()
    hamDir = args.hamDir
    spamDir= args.spamDir
    
    print("### Looking for ham files in",hamDir,
          "and spam files in", spamDir, "###")

    trainX,trainY,testX,testY = input_data(hamDir,spamDir,.1)
    
    np.savetxt("trainX.csv", trainX, delimiter=",")
    np.savetxt("trainY.csv", trainY, delimiter=",")
    np.savetxt("testX.csv", testX, delimiter=",")
    np.savetxt("testY.csv", testY, delimiter=",")
