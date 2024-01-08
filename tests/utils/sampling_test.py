""" Tests for sampling functions """

# Author: Mohamed Abouelsaadat
# License: MIT

import unittest
import unittest.mock
import numpy as np
from randoptma.utils.sampling import initialize_uniform


class TestUniformInitialization(unittest.TestCase):
    def setUp(self):
        self.patcher = unittest.mock.patch("numpy.random.default_rng")
        self.addCleanup(self.patcher.stop)
        self.feat_dict = {0: [0, 1], 1: (1, 2), 2: [2, 3]}
        self.rng = np.random.default_rng(None)
        self.feat_dict_rand = {
            feat: sorted(
                self.rng.choice(
                    a=10,
                    size=self.rng.integers(low=2, high=10),
                    replace=False,
                ).tolist()
            )
            if self.rng.choice([False, True])
            else (0, self.rng.integers(low=2, high=10))
            for feat in range(self.rng.integers(low=2, high=10))
        }

    def test_single_sample_size(self):
        self.failUnlessEqual(
            initialize_uniform(self.feat_dict_rand).shape,
            (len(self.feat_dict_rand),),
        )

    def test_single_sample(self):
        rng = self.patcher.start().return_value
        rng.choice.side_effect = [0, 2]
        rng.uniform.return_value = 0.5
        self.failUnlessEqual(
            initialize_uniform(self.feat_dict).tolist(), [0.0, 0.5, 2.0]
        )

    def test_multi_samples_size(self):
        size = self.rng.integers(low=10, high=20)
        self.failUnlessEqual(
            initialize_uniform(self.feat_dict_rand, size=size).shape,
            (
                size,
                len(self.feat_dict_rand),
            ),
        )

    def test_multi_samples(self):
        size = self.rng.integers(low=10, high=20)
        discrete_list0 = self.rng.choice(self.feat_dict[0], size=(1, size))
        discrete_list2 = self.rng.choice(self.feat_dict[2], size=(1, size))
        continuous_list1 = self.rng.uniform(
            low=self.feat_dict[1][0], high=self.feat_dict[1][1], size=size
        )
        rng = self.patcher.start().return_value
        rng.choice.side_effect = np.concatenate(
            (discrete_list0, discrete_list2), axis=0
        )
        rng.uniform.return_value = continuous_list1
        self.failUnlessEqual(
            initialize_uniform(self.feat_dict, size=size).tolist(),
            np.concatenate(
                (discrete_list0.T, continuous_list1[..., np.newaxis], discrete_list2.T),
                axis=1,
            ).tolist(),
        )
