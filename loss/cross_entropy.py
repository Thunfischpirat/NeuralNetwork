""" Implement cross entropy loss function. """
from activations.softmax import Softmax
import numpy as np

class CrossEntropyLoss():
    """ Cross entropy loss function. """
    def forward(self, y, y_hat):
        """ Forward pass of cross entropy loss function. """
        return -np.sum(y * np.log(y_hat))

    # Make forward method callable like CrossEntropy(y, y_hat)
    __call__ = forward

    def backward(self, y, y_hat):
        """ Backward pass of cross entropy loss function. """
        return -y / y_hat


class SMCrossEntropyLoss():
    """ cross entropy loss function with build in softmax application. """
    def forward(self, x, y):
        """ Forward pass of softmax cross entropy loss function. """
        self.y_hat = Softmax()(x)
        return -np.sum(y * np.log(self.y_hat))

    # Make forward method callable like SMCrossEntropy(x, y)
    __call__ = forward

    def backward(self, y, y_hat):
        """ Backward pass of softmax cross entropy loss function. """
        return y_hat - y
