"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import numpy as np


def singlepoint(parents, seed=None):
    rng = np.random.default_rng(seed)
    crossover_indx = rng.integers(len(parents[0]))
    offspring = np.empty([2, len(parents[0])])

    offspring[0, :crossover_indx] = parents[0, :crossover_indx]
    offspring[0, crossover_indx:] = parents[1, crossover_indx:]

    offspring[1, :crossover_indx] = parents[1, :crossover_indx]
    offspring[1, crossover_indx:] = parents[0, crossover_indx:]
    return offspring


def doublepoint(parents, seed=None):
    rng = np.random.default_rng(seed)
    crossover_indx1 = rng.integers(low=0, high=int(0.75 * len(parents[0])))
    crossover_indx2 = rng.integers(low=crossover_indx1, high=len(parents[0]))

    offspring = np.empty([2, len(parents[0])])
    offspring[0, :crossover_indx1] = parents[0, :crossover_indx1]
    offspring[0, crossover_indx1:crossover_indx2] = parents[
        1, crossover_indx1:crossover_indx2
    ]
    offspring[0, crossover_indx2:] = parents[0, crossover_indx2:]

    offspring[1, :crossover_indx1] = parents[1, :crossover_indx1]
    offspring[1, crossover_indx1:crossover_indx2] = parents[
        0, crossover_indx1:crossover_indx2
    ]
    offspring[1, crossover_indx2:] = parents[1, crossover_indx2:]
    return offspring


def uniform(parents, seed=None):
    rng = np.random.default_rng(seed)
    offspring = np.empty([2, len(parents[0])])
    for feature in range(len(parents[0])):
        offspring[:, feature] = rng.choice(a=parents[:, feature], size=2, replace=False)
    return offspring
