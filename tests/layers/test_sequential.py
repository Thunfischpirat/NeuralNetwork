""" Unit-tests for sequential module. """
from unittest import TestCase
import numpy as np
from activations.relu import ReLU
from activations.softmax import Softmax
from layers.linear import LinearLayer
from layers.sequential import Sequential


class TestSequential(TestCase):
    def setUp(self):
        """Set up the Sequential model."""
        self.model = Sequential(LinearLayer(14, 10),
                                ReLU(),
                                LinearLayer(10, 8),
                                Softmax())
    def test_forward(self):
        """Test the forward pass of the Sequential model."""
        data = np.random.rand(14)
        output = self.model.forward(data)
        # Check shape
        self.assertEqual(output.shape, (8,))
        # Check activation of last layer
        self.assertTrue(np.allclose(np.sum(output), 1))
