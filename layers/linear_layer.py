import numpy as np


class LinearLayer():
    def __init__(self, input_dim, output_dim):
        self.W = np.random.randn(output_dim, input_dim)
        self.b = np.random.randn(output_dim)

    def forward(self, x):
        return np.dot(self.W,x) + self.b









