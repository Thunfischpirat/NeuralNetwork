""" Unit-tests for activations.relu module. """
from unittest import TestCase
import numpy as np

from activations.relu import ReLU, LeakyReLU


class TestReLU(TestCase):
    """Test the ReLU activation function."""

    def setUp(self):
        """Set up the ReLU activation function."""
        self.relu = ReLU()

    def test_forward(self):
        """Test the forward pass of the ReLU activation function."""
        # Test shape
        data = np.array([-1, 0, 1])
        output = self.relu.forward(data)
        # Validate output shape.
        self.assertEqual(output.shape, (3,))
        # Validate output values.
        expected_output = [0, 0, 1]
        self.assertTrue(np.allclose(output, expected_output))

    def test_backward(self):
        """Test the backward pass of the ReLU activation function."""
        # Test shape
        data = np.array([-1, 0, 1])
        output = self.relu.backward(data)
        # Validate output shape.
        self.assertEqual(output.shape, (3,))
        # Validate output values.
        expected_output = [0, 0, 1]
        self.assertTrue(np.allclose(output, expected_output))


class TestLeakyReLU(TestCase):
    """Test the LeakyReLU activation function."""

    def setUp(self):
        """Set up the LeakyReLU activation function."""
        self.leaky_relu = LeakyReLU()

    def test_forward(self):
        """Test the forward pass of the LeakyReLU activation function."""
        # Test shape
        data = np.array([-1, 0, 1])
        output = self.leaky_relu.forward(data)
        # Validate output shape.
        self.assertEqual(output.shape, (3,))
        # Validate output values.
        expected_output = [-0.01, 0, 1]
        self.assertTrue(np.allclose(output, expected_output))

    def test_backward(self):
        """Test the backward pass of the LeakyReLU activation function."""
        # Test shape
        data = np.array([-1, 0, 1])
        output = self.leaky_relu.backward(data)
        # Validate output shape.
        self.assertEqual(output.shape, (3, ))
        # Validate output values.
        expected_output = [0.01, 0.01, 1]
        self.assertTrue(np.allclose(output, expected_output))
