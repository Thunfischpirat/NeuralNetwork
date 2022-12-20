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

    def backward(self, grad_output):
        """ Backward pass of Sequential model. """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def update_params(self, learning_rate):
        """ Update parameters of Sequential model. """
        for layer in self.layers:
            if layer.trainable:
                layer.update_params(learning_rate)