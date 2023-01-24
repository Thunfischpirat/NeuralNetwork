from unittest import TestCase

import numpy as np

from layers.conv2d import Conv2DLayer


class TestConv2DLayer(TestCase):
    def setUp(self):
        """Set up the test."""
        # Output of the first convolutional layer computed by hand.
        self.expected_output = np.array(
            [
                [
                    [4.0, 7.0, 7.0, 6.0, 3.0],
                    [7.0, 11.0, 12.0, 9.0, 5.0],
                    [7.0, 11.0, 14.0, 12.0, 8.0],
                    [8.0, 12.0, 12.0, 10.0, 6.0],
                    [5.0, 8.0, 7.0, 7.0, 4.0],
                ],
                [
                    [4.0, 7.0, 7.0, 6.0, 3.0],
                    [7.0, 11.0, 12.0, 9.0, 5.0],
                    [7.0, 11.0, 14.0, 12.0, 8.0],
                    [8.0, 12.0, 12.0, 10.0, 6.0],
                    [5.0, 8.0, 7.0, 7.0, 4.0],
                ],
            ]
        )

    def test_forward(self):

        np.random.seed(32)
        img = np.random.randint(0, 2, (2, 5, 5))
        conv1 = Conv2DLayer(2, 2, 3, stride=1, padding=1)
        # Set weights and biases to 1 for easier testing
        conv1.W = np.ones_like(conv1.W)
        conv1.b = np.ones_like(conv1.b)

        out = conv1(img)
        self.assertEqual(out.shape, (2, 5, 5))
        self.assertTrue(np.allclose(out, self.expected_output))

        # Apply another layer of convolutions.
        conv2 = Conv2DLayer(2, 4, 3, stride=2, padding=1)
        conv2.W = np.ones_like(conv2.W)
        conv2.b = np.ones_like(conv2.b)
        out = conv2(out)
        self.assertEqual(out.shape, (4, 3, 3))

