""" """

# Author: Mohamed Abouelsaadat
# License: MIT

import math
import numpy as np
from randoptma.param_tuning.gridsearch import gridsearch
import randoptma.algorithms.genetic.algo as genetic_algo


def euclidean_distance(pointA, pointB):
    return math.sqrt((pointA[0] - pointB[0]) ** 2 + (pointA[1] - pointB[1]) ** 2)


def TSP(input_x, positions):
    """Cities are visited based on their values ascending order,
    where cities with equal values are visited in their index order"""
    total = 0
    order = np.argsort(input_x, kind="mergesort")
    for itr in range(len(order)):
        total += euclidean_distance(
            positions[order[itr], :], positions[order[itr - 1], :]
        )
    return -total


ENTRY_LENGTH = 10
positions = np.random.default_rng().uniform(0, 100, (ENTRY_LENGTH, 2))


def problem_eval_function(input):
    return TSP(input, positions)


def problem_feat_dict():
    return {feat: list(range(ENTRY_LENGTH)) for feat in range(ENTRY_LENGTH)}


gridsearch(
    genetic_algo.optimize,
    problem_feat_dict(),
    problem_eval_function,
    optimizer_params={"pop_size": [500, 1000, 1500], "replaced_frac": [0.1, 0.2, 0.4]},
    n_runs=2,
    n_jobs=5,
    verbose=True,
)
