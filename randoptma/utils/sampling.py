"""A group of functions to generate a list of samples given a search space or a new sample sample given a search space and a sample from this space, it could be used to initialize a sample popuation or as a mutation or a neighbor for the given sample"""

# Author: Mohamed Abouelsaadat
# License: MIT

import numpy as np


def new_seed(rng):
    return rng.integers(1e5)


def initialize_uniform(feat_dict: dict, size: int = None, seed: int = None):
    rng = np.random.default_rng(seed)
    sample_X = (
        np.empty([size, len(feat_dict)]) if size else np.empty([1, len(feat_dict)])
    )
    for key, values in feat_dict.items():
        if _is_discrete_format(values):
            sample_X[:, key] = rng.choice(a=values, size=size)
        elif _is_continuous_format(values):
            sample_X[:, key] = rng.uniform(
                low=min(values),
                high=max(values),
                size=size,
            )
        else:
            raise TypeError(
                f"Value of the key <{key}> in features dictionary is not correct, "
                "use either tuple for continous features or list for discrete features"
            )
    return sample_X if size else sample_X[0]


def one_variable_uniform(feat_dict: dict, sample_x: list, seed: int = None):
    rng = np.random.default_rng(seed)
    new_sample_x = sample_x.copy()
    feat_indx = rng.integers(len(feat_dict))
    if _is_discrete_format(feat_dict[feat_indx]):
        new_sample_x[feat_indx] = rng.choice(
            a=[val for val in feat_dict[feat_indx] if val != sample_x[feat_indx]],
            size=None,
        )
    elif _is_continuous_format(feat_dict[feat_indx]):
        new_sample_x[feat_indx] = rng.uniform(
            low=min(feat_dict[feat_indx]),
            high=max(feat_dict[feat_indx]),
            size=None,
        )
    else:
        raise TypeError(
            f"Value of the key <{feat_indx}> in features dictionary is not correct, "
            "use either tuple for continous features or list for discrete features"
        )
    return new_sample_x


def one_variable_triangular(feat_dict: dict, sample_x: list, seed: int = None):
    rng = np.random.default_rng(seed)
    new_sample_x = sample_x.copy()
    feat_indx = rng.integers(len(feat_dict))
    if _is_discrete_format(feat_dict[feat_indx]):
        index = new_index = feat_dict[feat_indx].index(sample_x[feat_indx])
        while index == new_index:
            new_index = rng.triangular(
                left=0,
                mode=index,
                right=len(feat_dict[feat_indx]) - 1,
                size=None,
            )
            new_index = round(new_index)
        new_sample_x[feat_indx] = feat_dict[feat_indx][new_index]
    elif _is_continuous_format(feat_dict[feat_indx]):
        min_value = min(feat_dict[feat_indx])
        max_value = max(feat_dict[feat_indx])
        full_width = max_value - min_value
        while (
            abs(new_sample_x[feat_indx] - sample_x[feat_indx])
            < (rng.random() * 0.09 + 0.01) * full_width
        ):
            new_sample_x[feat_indx] = rng.triangular(
                left=min_value,
                mode=sample_x[feat_indx],
                right=max_value,
                size=None,
            )
    else:
        raise TypeError(
            f"Value of the key <{feat_indx}> in features dictionary is not correct, "
            "use either tuple for continous features or list for discrete features"
        )
    return new_sample_x


def one_variable_triangular_rounded(feat_dict: dict, sample_x: list, seed: int = None):
    rng = np.random.default_rng(seed)
    new_sample_x = sample_x.copy()
    feat_indx = rng.integers(len(feat_dict))
    if _is_discrete_format(feat_dict[feat_indx]):
        full_width = len(feat_dict[feat_indx])
        half_width = full_width / 2.0
        index = new_index = feat_dict[feat_indx].index(sample_x[feat_indx])
        while index == new_index:
            new_index = rng.triangular(
                left=index - half_width,
                mode=index,
                right=index + half_width,
                size=None,
            )
            new_index = round(new_index) % full_width
        new_sample_x[feat_indx] = feat_dict[feat_indx][new_index]
    elif _is_continuous_format(feat_dict[feat_indx]):
        min_value = min(feat_dict[feat_indx])
        max_value = max(feat_dict[feat_indx])
        full_width = max_value - min_value
        half_width = full_width / 2.0
        while (
            abs(new_sample_x[feat_indx] - sample_x[feat_indx])
            < (rng.random() * 0.09 + 0.01) * full_width
        ):
            new_value = rng.triangular(
                left=sample_x[feat_indx] - half_width,
                mode=sample_x[feat_indx],
                right=sample_x[feat_indx] + half_width,
                size=None,
            )
            new_sample_x[feat_indx] = new_value % full_width
    else:
        raise TypeError(
            f"Value of the key <{feat_indx}> in features dictionary is not correct, "
            "use either tuple for continous features or list for discrete features"
        )
    return new_sample_x


def _is_discrete_format(available_values):
    return type(available_values) is list and len(np.unique(available_values)) > 1


def _is_continuous_format(available_values):
    return (
        type(available_values) is tuple
        and len(available_values) > 1
        and min(available_values) < max(available_values)
    )
