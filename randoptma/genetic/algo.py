"""Genetic Algorithm Optimization based on Tom Mitchell's book: Machine Learning, Chapter 9"""

# Author: Mohamed Abouelsaadat
# License: MIT

import math
import warnings
import numpy as np
from multiprocess import Pool
from .crossover import singlepoint
from ..utils.sampling import (
    new_seed,
    initialize_uniform,
    one_variable_triangular_rounded,
)


def optimize(
    feat_dict,
    eval_func,
    crossover_func=singlepoint,
    pop_size=1000,
    replaced_frac=0.4,
    mutation_rate=0.2,
    n_iter_no_change=10,
    max_iter=1000,
    seed=None,
    verbose=False,
    n_jobs=None,
):
    """Implementation of Genetic Algorithm based on Mitchell, T. M. (1997). Machine learning (Vol. 1). McGraw-hill New York.

    Params
    ------
    feat_dict: dictionary with keys representing features indices, and values representing valid values.
        discrete ex : [0,1,2,3,4]
        continuous ex : (-1, 1)
    eval_func: evaluation function used to measure performance of each sample.
    crossover_func: the function to use in crossover
    pop_size: positive integer value representing the population size to be used
    replaced_frac: fraction of population to be replaced by offsprings (0..1)
    mutation_rate: mutation rate to be applied to new population (0..1)
    n_iter_no_change: number of iterations with no change in best score to determine convergence
    max_iter: total max iterations allowed
    seed: random seed to be used in random numbers generation, if None an arbitrary random seed is chosen
    verbose: boolean value to switch on/off the printing of each iteration results
    n_jobs: number of processes to spawn, must be a positive integer or None for max

    Return
    ------
    sample with highest score, highest score
    """
    sample_X = None
    rng = np.random.default_rng(seed)
    replace_size = int((replaced_frac * pop_size) // 2 * 2)
    keep_size = pop_size - replace_size

    _iter_ = iter(range(max_iter))
    for iteration in _iter_:
        if sample_X is None:
            # Generate a uniform sample
            sample_X = initialize_uniform(
                feat_dict=feat_dict, size=pop_size, seed=new_seed(rng)
            )
            evals, best_index, median_index = _get_evals(sample_X, eval_func, n_jobs)
        if verbose:
            print(
                "\niteration:",
                iteration,
                "\nbest score:",
                evals[best_index],
                "\nbest sample:",
                "-".join(str(feature_val) for feature_val in sample_X[best_index]),
            )
        # Build new population
        is_new_sample = False
        for _ in range(n_iter_no_change):
            new_sample_X = np.empty([pop_size, len(feat_dict)])
            new_sample_X[:keep_size, :] = _next_keep_population(
                sample_X,
                evals,
                keep_size,
                new_seed(rng),
            )
            new_sample_X[keep_size:, :] = _produce_offsprings(
                sample_X,
                evals,
                replace_size,
                lambda parents: crossover_func(parents, new_seed(rng)),
                new_seed(rng),
                n_jobs,
            )
            original_new_sample_X = new_sample_X.copy()
            _mutate_population(
                new_sample_X, feat_dict, mutation_rate, new_seed(rng), n_jobs
            )
            # Calculate sample evals
            new_evals, new_best_index, new_median_index = _get_evals(
                new_sample_X, eval_func
            )
            if (
                new_evals[new_best_index] > evals[best_index]
                or math.isclose(new_evals[new_best_index], evals[best_index])
                and new_evals[new_median_index] > evals[median_index]
            ):
                sample_X, evals, best_index, median_index, is_new_sample = (
                    new_sample_X,
                    new_evals,
                    new_best_index,
                    new_median_index,
                    True,
                )
                break
            elif next(_iter_, None) is None:
                warnings.warn(
                    f"Stochastic Optimizer: Maximum iterations ({max_iter}) reached and the optimization hasn't converged yet.",
                    RuntimeWarning,
                )
                break
        if is_new_sample == False:
            break
    return sample_X[best_index], evals[best_index]


# Helper functions
def _get_evals(sample_X, eval_func, n_jobs=None):
    with Pool(n_jobs) as p:
        evals = np.asarray(p.map(eval_func, sample_X))
    order = np.argsort(evals)
    best_index = order[len(sample_X) - 1]
    median_index = order[len(sample_X) // 2]
    return evals, best_index, median_index


def _next_keep_population(
    sample_X,
    evals,
    keep_size,
    seed=None,
):
    rng = np.random.default_rng(seed)
    _evals_ = evals - np.min(evals)  # make minimum zero
    _evals_ += 0.01 * np.max(_evals_)  # add minute value
    keep_population = rng.choice(
        a=sample_X,
        size=keep_size,
        p=_evals_ / np.sum(_evals_),
    )
    return keep_population


def _produce_offsprings(
    sample_X,
    evals,
    offspring_size,
    crossover_func,
    seed=None,
    n_jobs=None,
):
    rng = np.random.default_rng(seed)
    _evals_ = evals - np.min(evals)  # make minimum zero
    _evals_ += 0.01 * np.max(_evals_)  # add minute value
    parents_pairs = rng.choice(
        a=sample_X,
        size=(offspring_size // 2, 2),
        p=_evals_ / np.sum(_evals_),
    )
    with Pool(n_jobs) as p:
        offsprings = np.asarray(p.map(crossover_func, parents_pairs))
    return offsprings.reshape(-1, offsprings.shape[-1])  # collapse one dimension


def _mutate_population(input_X, feat_dict, mutation_rate, seed=None, n_jobs=None):
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        a=len(input_X),
        size=int(mutation_rate * len(input_X)),
        replace=False,
    )
    indices_seeds = np.concatenate(
        (indices[:, np.newaxis], new_seed(rng, size=len(indices))[:, np.newaxis]),
        axis=1,
    )
    with Pool(n_jobs) as p:
        input_X[indices] = np.asarray(
            p.map(
                lambda input: one_variable_triangular_rounded(
                    feat_dict, input_X[input[0]], input[1]
                ),
                indices_seeds,
            )
        )
