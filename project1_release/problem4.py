# -------------------------------------------------------------------------
'''
    Problem 4: compute sigmoid(Z), the loss function, and the gradient.
    This is the vectorized version that handle multiple training examples X.

    20/100 points
'''

import numpy as np # linear algebra
from scipy.sparse import diags
from scipy.sparse import csr_matrix

def linear(theta, X):
    '''
    theta: (n+1) x 1 column vector of model parameters
    x: (n+1) x m matrix of m training examples, each with (n+1) features.
    :return: inner product between theta and x
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return np.dot(theta.T,X)
    #########################################

def sigmoid(Z):
    '''
    Z: 1 x m vector. <theta, X>
    :return: A = sigmoid(Z)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return 1 / (1 + np.exp(-Z))
    #########################################

def loss(A, Y):
    '''
    A: 1 x m, sigmoid output on m training examples
    Y: 1 x m, labels of the m training examples

    You must use the sigmoid function you defined in *this* file.

    :return: mean negative log-likelihood loss on m training examples.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return (-(np.inner(np.log(A),Y) + np.inner(np.log(1 - A), (1 - Y))))/Y.shape[1]
    #########################################

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

def dtheta(Z, X, Y):
    '''
    Z: 1 x m vector. <theta, X>
    X: (n+1) x m, m example feature vectors.
    Y: 1 x m, label of X
    :return: d x 1, mean of the gradient of the negative log-likelihood loss on all samples wrt w.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return np.dot(X,dZ(Z, Y).T)/len(X[0])
    #########################################

def Hessian(Z, X):
    '''
    Compute the Hessian matrix on m training examples.
    Z: 1 x m vector. <theta, X>
    X: (n+1) x m, m example feature vectors.
    :return: the Hessian matrix of the negative log-likelihood loss wrt theta
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    Sigma=np.diag(np.squeeze(sigmoid(Z)*(1-sigmoid(Z))))
    H=np.dot(np.dot(X,Sigma),X.T)
    return H
    #########################################
