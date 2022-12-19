""" Implements linear layer of a neural network. """
import numpy as np


class LinearLayer():
    def __init__(self, input_dim, output_dim):
        """ Initialize LinearLayer. """
        self.W = np.random.randn(output_dim, input_dim)
        self.b = np.random.randn(output_dim)

    def forward(self, x):
        """ Forward pass of LinearLayer. """
        self.input = x
        return np.dot(self.W,x) + self.b

    # Make forward method callable like LinearLayer(x)
    __call__ = forward

    def backward(self, grad_output):
        """ Backward pass of LinearLayer. """
        x = self.input
        self.grad_W = np.outer(grad_output, x)
        self.grad_b = grad_output
        return np.dot(self.W.T, grad_output)









