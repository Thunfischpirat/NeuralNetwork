""" Implements softmax activation function. """

import numpy as np

class Softmax():
    """ Softmax activation function. """
    def forward(self, x):
        """ Forward pass of Softmax activation function. """
        z = x - np.max(x)
        exp = np.exp(z)
        probs = exp / np.sum(exp)
        return probs

    # Make forward method callable lik Softmax(x)
    __call__ = forward

    def backward(self, y):
        """ Backward pass of Softmax activation function. """
        jacobian = np.diag(y) - np.outer(y, y)
        return jacobian
