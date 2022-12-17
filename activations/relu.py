""" Implements different relu based activation functions. """

import numpy as np

class ReLU():
    """ ReLU activation function. """
    def forward(self, x):
        """ Forward pass of ReLU activation function. """
        return np.maximum(0, x)

    def backward(self, x):
        """ Backward pass of ReLU activation function. """
        return np.where(x > 0, 1, 0)


class LeakyReLU():
    """ Leaky ReLU activation function. """
    def __init__(self, alpha: float=0.01):
        """ Initialize LeakyReLU activation function. """
        self.alpha = alpha

    def forward(self, x):
        """ Forward pass of LeakyReLU activation function. """
        return np.maximum(x, self.alpha * x)

    def backward(self, x):
        """ Backward pass of LeakyReLU activation function. """
        return np.where(x > 0, 1, self.alpha)