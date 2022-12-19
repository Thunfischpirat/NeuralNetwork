from unittest import TestCase
from layers.linear import LinearLayer
import numpy as np


class TestLinearLayer(TestCase):
    def setUp(self):
        self.layer = LinearLayer(3, 2)
        self.layer.W = np.array([[1, 2, 1], [2, 2, 2]])
        self.layer.b = np.array([1, 2])

    def test_forward(self):
        # Test shape
        data = np.array([1,2,3])
        output = self.layer.forward(data)
        # Validate output shape.
        self.assertEqual(output.shape, (2,))
        # Validate output values.
        expected_output = np.array([9, 14])
        self.assertTrue(np.allclose(output, expected_output))
