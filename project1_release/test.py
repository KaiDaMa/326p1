import numpy as np
import math
def linear(theta, X):
    '''
    theta: (n+1) x 1 column vector of model parameters
    x: (n+1) x m matrix of m training examples, each with (n+1) features.
    :return: inner product between theta and x
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return np.matmul(theta.T,X)
def sigmoid(Z):
    '''
    Z: 1 x m vector. <theta, X>
    :return: A = sigmoid(Z)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return 1 / (1 + np.exp(-Z))
def dZ(Z, Y):
    '''
    Z: 1 x m vector. <theta, X>
    Y: 1 x m, label of X

    You must use the sigmoid function you defined in *this* file.

    :return: 1 x m, the gradient of the negative log-likelihood loss on all samples wrt z.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return sigmoid(Z)-Y
    #########################################
X = np.array([[1,1], [2, 1], [2, 3], [1, 3]])
Y = np.array([[1, 0]])
print(Y.shape)
