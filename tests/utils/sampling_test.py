""" Tests for sampling functions """

# Author: Mohamed Abouelsaadat
# License: MIT

import unittest
import unittest.mock
import numpy as np


def test(seed=12345):
    rng = np.random.default_rng(seed)
    return rng.integers(10)


class Test(unittest.TestCase):
    def setUp(self):
        patcher = unittest.mock.patch("numpy.random.default_rng")
        self.addCleanup(patcher.stop)
        self.rng = patcher.start().return_value

    def test_single_return_value(self):
        self.rng.integers.return_value = 5
        assert test() == 5
        assert test() == 5

    def test_multiple_return_value(self):
        self.rng.integers.side_effect = [1, 2, 3]
        assert test() == 1
        assert test() == 2
        assert test() == 3
