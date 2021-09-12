# -------------------------------------------------------------------------
'''
    Problem 2: reading data set from a file, and then split them into training, validation and test sets.

    The functions for handling data

    20/100 points
'''

import numpy as np # for linear algebra

def loadData():
    '''
        Read all labeled examples from the text files.
        Note that the data/X.txt has a row for a feature vector for intelligibility.

        n: number of features
        m: number of examples.

        :return: X: numpy.ndarray. Shape = [n, m]
                y: numpy.ndarray. Shape = [m, ]
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    X = np.loadtxt("data/X.txt").transpose()
    y = np.loadtxt("data/y.txt").transpose()
    return X, y
    #########################################


def appendConstant(X):
    '''
    Appending constant "1" to the beginning of each training feature vector.
    X: numpy.ndarray. Shape = [n, m]
    :return: return the training samples with the appended 1. Shape = [n+1, m]
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    ones=np.ones((1,np.shape(X)[1]))
    X = np.append(ones, X, axis = 0)
    return X
    #########################################


def splitData(X, y, train_ratio = 0.8):
    '''
	X: numpy.ndarray. Shape = [n+1, m]
	y: numpy.ndarray. Shape = [m, ]
    split_ratio: the ratio of examples go into the Training, Validation, and Test sets.
    Split the whole dataset into Training, Validation, and Test sets.
    :return: return Training, Validation, and Test sets.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    border = int(np.shape(X)[1]*train_ratio)
    tr_X, test_X =X[:,:border],X[:,border:]
    tr_y, test_y =y[:border],y[border:]
    return (tr_X, tr_y), (test_X, test_y)
    #########################################
