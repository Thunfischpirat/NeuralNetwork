from unittest import TestCase
from layers.linear_layer import LinearLayer
import numpy as np


class TestLinearLayer(TestCase):
    def setUp(self):
        np.random.seed(32)
        self.layer = LinearLayer(3, 2)
        self.layer.W = np.random.randint(0, 5, self.layer.W.shape)
        self.layer.b = np.random.randint(0, 5, self.layer.b.shape)

    def test_forward(self):
        # Test shape
        data = np.random.randint(0, 9, (4, 3))
        output = self.layer.forward(data)
        # Validate output shape.
        self.assertEqual(output.shape, (4, 2))
        # Validate output values.
        expected_output = np.array([[24, 20], [43, 32], [24, 16], [15, 10]])
        self.assertTrue(np.allclose(output, expected_output))
