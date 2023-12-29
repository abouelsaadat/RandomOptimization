""" Supporting library which contains various functions for entropy and mutual information calculations """

# Author: Mohamed Abouelsaadat
# License: MIT

import numpy as np


def entropy(input_X):
    _, counts = np.unique(input_X, return_counts=True)
    probabilities = counts / input_X.shape[0]
    return -np.sum(probabilities * np.log2(probabilities))


def joint_entropy(input_X, input_Y):
    input_XY = np.concatenate((input_X[np.newaxis].T, input_Y[np.newaxis].T), axis=1)
    _, counts = np.unique(input_XY, return_counts=True, axis=0)
    probabilities = counts / input_XY.shape[0]
    return -np.sum(probabilities * np.log2(probabilities))


def conditional_entropy(input_X, input_Y):
    input_XY = np.concatenate((input_X[np.newaxis].T, input_Y[np.newaxis].T), axis=1)
    unique_values, counts = np.unique(input_XY, return_counts=True, axis=0)
    probabilities = counts / input_XY.shape[0]
    y_unique_values, y_counts = np.unique(input_XY[:, 1], return_counts=True, axis=0)
    # source : Example 2: Print First Index Position of Several Values
    # (https://www.geeksforgeeks.org/how-to-find-the-index-of-value-in-numpy-array/)
    sorter = np.argsort(y_unique_values)
    indices = sorter[
        np.searchsorted(y_unique_values, unique_values[:, 1], sorter=sorter)
    ]
    y_probabilities = y_counts[indices] / input_XY.shape[0]
    return -np.sum(probabilities * np.log2(probabilities / y_probabilities))


def mutual_information(input_X, input_Y):
    return entropy(input_X) - conditional_entropy(input_X, input_Y)
