"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import unittest
import numpy as np
from randoptma.mimic.utils.information import (
    entropy,
    joint_entropy,
    conditional_entropy,
    mutual_information,
)


class TestEntropyMethods(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng()
        size = rng.integers(low=50, high=100)
        self.arr1 = rng.choice(a=10, size=size)
        self.arr2 = rng.choice(a=10, size=size)

    def test_commutativejointentropy(self):
        self.assertAlmostEqual(
            joint_entropy(self.arr1, self.arr2),
            joint_entropy(self.arr2, self.arr1),
        )

    def test_jointentropyrelation(self):
        self.assertAlmostEqual(
            joint_entropy(self.arr1, self.arr2),
            conditional_entropy(self.arr1, self.arr2) + entropy(self.arr2),
        )
        self.assertAlmostEqual(
            joint_entropy(self.arr2, self.arr1),
            conditional_entropy(self.arr2, self.arr1) + entropy(self.arr1),
        )

    def test_mutualinformationcommutativecalculation(self):
        mutualinformation = entropy(self.arr1) - conditional_entropy(
            self.arr1, self.arr2
        )
        self.assertAlmostEqual(
            mutualinformation,
            entropy(self.arr2) - conditional_entropy(self.arr2, self.arr1),
        )
        self.assertAlmostEqual(
            mutualinformation,
            entropy(self.arr1)
            + entropy(self.arr2)
            - joint_entropy(self.arr1, self.arr2),
        )
        self.assertAlmostEqual(
            mutualinformation,
            joint_entropy(self.arr1, self.arr2)
            - conditional_entropy(self.arr1, self.arr2)
            - conditional_entropy(self.arr2, self.arr1),
        )


class TestMutualInformationMethods(unittest.TestCase):
    def setUp(self):
        self.arr1 = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])
        self.arr2 = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])

    def test_entropy(self):
        self.assertAlmostEqual(entropy(self.arr1[:, 0]), 1.0)
        self.assertAlmostEqual(entropy(self.arr1[:, 1]), 1.0)
        self.assertAlmostEqual(entropy(self.arr2[:, 0]), 1.0)
        self.assertAlmostEqual(entropy(self.arr2[:, 1]), 1.0)

    def test_jointentropy(self):
        self.assertAlmostEqual(joint_entropy(self.arr1[:, 0], self.arr1[:, 1]), 1.0)
        self.assertAlmostEqual(joint_entropy(self.arr2[:, 0], self.arr2[:, 1]), 2.0)

    def test_conditionalentropy(self):
        self.assertAlmostEqual(
            conditional_entropy(self.arr1[:, 0], self.arr1[:, 1]), 0.0
        )
        self.assertAlmostEqual(
            conditional_entropy(self.arr2[:, 0], self.arr2[:, 1]), 1.0
        )

    def test_mutualinformation(self):
        self.assertAlmostEqual(
            mutual_information(self.arr1[:, 0], self.arr1[:, 1]), 1.0
        )
        self.assertAlmostEqual(
            mutual_information(self.arr2[:, 0], self.arr2[:, 1]), 0.0
        )


if __name__ == "__main__":
    unittest.main()
