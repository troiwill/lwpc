# Generated using ChatGPT

import unittest
import numpy as np
from lwpc.utils import se3inv


class TestSE3Inv(unittest.TestCase):
    def test_input_type(self):
        with self.assertRaises(TypeError):
            se3inv(1)

    def test_input_shape(self):
        with self.assertRaises(ValueError):
            se3inv(np.zeros((2, 2, 2)))

    def test_last_row(self):
        with self.assertRaises(ValueError):
            se3inv(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1]]))

    def test_inverse(self):
        pose = np.array([[1, 0, 0, 1], [0, 1, 0, 2], [0, 0, 1, 3], [0, 0, 0, 1]])
        pose_inv = np.array([[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, -3], [0, 0, 0, 1]])
        self.assertTrue(np.allclose(se3inv(pose), pose_inv))


if __name__ == "__main__":
    unittest.main()
