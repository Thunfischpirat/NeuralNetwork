""" Implements 2d convolutional layer of a neural network. """

import numpy as np


class Conv2DLayer:
    def __init__(
        self, input_channels, output_channels, kernel_size, stride=1, padding=0
    ):
        """Initialize Conv2DLayer with He-method."""
        self.W = np.random.randn(
            output_channels, input_channels, kernel_size, kernel_size
        ) * np.sqrt(2 / (input_channels * kernel_size * kernel_size))
        self.b = np.random.randn(output_channels) * np.sqrt(
            2 / (input_channels * kernel_size * kernel_size)
        )
        self.trainable = True
        self.stride = stride
        self.padding = padding
        self.output_channels = output_channels
        self.input_channels = input_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        """Forward pass of Conv2DLayer."""
        self.h_out = (
            x.shape[1] + 2 * self.padding - self.kernel_size
        ) // self.stride + 1
        self.w_out = (
            x.shape[2] + 2 * self.padding - self.kernel_size
        ) // self.stride + 1

        x_padded = np.pad(
            x,
            ((0, 0), (self.padding, self.padding), (self.padding, self.padding)),
            constant_values=0,
        )

        self.input = x_padded

        Z = np.zeros((self.output_channels, self.h_out, self.w_out))
        for k in range(self.output_channels):
            for i in range(self.h_out):
                for j in range(self.w_out):
                    Z[k, i, j] = (
                        np.sum(
                            x_padded[
                                :,
                                i * self.stride : i * self.stride + self.kernel_size,
                                j * self.stride : j * self.stride + self.kernel_size,
                            ]
                            * self.W[k, :, :, :]
                        )
                        + self.b[k]
                    )

        return Z

    # Make forward method callable like Conv2DLayer(x)
    __call__ = forward

    def backward(self, grad_output):
        """Backward pass of Conv2DLayer."""
        self.grad_W = np.zeros(self.W.shape)
        self.grad_b = np.zeros(self.b.shape)

    def update_params(self, learning_rate):
        """Update parameters of Conv2DLayer."""
        self.W -= learning_rate * self.grad_W
        self.b -= learning_rate * self.grad_b
