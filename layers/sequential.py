""" Implements sequential model."""

import numpy as np

class Sequential():
    """ Sequential model. """
    def __init__(self, *layers):
        """ Initialize Sequential model. """
        self.layers = list(layers)

    def forward(self, x):
        """ Forward pass of Sequential model. """
        for layer in self.layers:
            x = layer(x)
        return x

    # Make forward method callable like Sequential(x)
    __call__ = forward

    def backward(self, x, grad_output):
        """ Backward pass of Sequential model. """
        for layer in reversed(self.layers):
            grad_output = layer.backward(x, grad_output)
        return grad_output

    def params(self):
        """ Get parameters of Sequential model. """
        result = []
        for layer in self.layers:
            result.extend(layer.params())
        return result

    def grad_params(self):
        """ Get gradient of parameters of Sequential model. """
        result = []
        for layer in self.layers:
            result.extend(layer.grad_params())
        return result

    def update_params(self, learning_rate):
        """ Update parameters of Sequential model. """
        for layer in self.layers:
            layer.update_params(learning_rate)

    def train(self):
        """ Set model to training mode. """
        for layer in self.layers:
            layer.train