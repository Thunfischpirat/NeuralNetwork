from unittest import TestCase

import numpy as np

from activations.softmax import Softmax


class TestSoftmax(TestCase):

    def setUp(self) -> None:
        self.data_in = np.random.rand(14)
        self.data_out = np.array([1/2, 1/4, 1/4])
        self.layer = Softmax()
    def test_forward(self):
        out = self.layer.forward(self.data_in)
        self.assertTrue(np.allclose(np.sum(out), 1))

    def test_backward(self):
        grad_out = self.layer.backward(self.data_out)
        expected_out = np.array([ 0.0625,  -0.03125, -0.03125])
        self.assertTrue(np.allclose(grad_out, expected_out))