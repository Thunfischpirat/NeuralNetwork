""" Implements different relu based activation functions. """

import numpy as np

class ReLU():
    """ ReLU activation function. """
    def forward(self, x):
        """ Forward pass of ReLU activation function. """
        self.input = x
        return np.maximum(0, x)

    # Make forward method callable like ReLU(x)
    __call__ = forward

    def backward(self, grad_output):
        """ Backward pass of ReLU activation function. """
        return grad_output * np.where(self.input > 0, 1, 0)


class LeakyReLU():
    """ Leaky ReLU activation function. """
    def __init__(self, alpha: float=0.01):
        """ Initialize LeakyReLU activation function. """
        self.alpha = alpha

    def forward(self, x):
        """ Forward pass of LeakyReLU activation function. """
        self.input = x
        return np.maximum(x, self.alpha * x)

    # Make forward method callable like LeakyReLU(x)
    __call__ = forward

    def backward(self, grad_output):
        """ Backward pass of LeakyReLU activation function. """
        return grad_output * np.where(self.input > 0, 1, self.alpha)