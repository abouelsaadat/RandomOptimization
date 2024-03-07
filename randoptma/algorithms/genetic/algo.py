"""Genetic Algorithm Optimization based on Tom Mitchell's book: Machine Learning, Chapter 9"""

# Author: Mohamed Abouelsaadat
# License: MIT

import math
import warnings
import numpy as np
from .crossover import singlepoint
from ..utils.sampling import (
    new_seed,
    initialize_uniform,
    one_variable_uniform,
)


def optimize(
    feat_dict,
    eval_func,
    crossover_func=singlepoint,
    pop_size=1000,
    replaced_frac=0.4,
    mutation_rate=0.2,
    n_iter_no_change=10,
    max_iter=int(1e10),
    epsilon=1e-3,
    seed=None,
    verbose=False,
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
    epsilon: smallest change taken into account as improvement
    seed: random seed to be used in random numbers generation, if None an arbitrary random seed is chosen
    verbose: boolean value to switch on/off the printing of each iteration results

    Return
    ------
    sample with highest score, highest score, array of iteration number vs score, number of function evaluations per iteration
    """
    sample_X = None
    score_per_iter = list()
    fevals_per_iter = pop_size
    rng = np.random.default_rng(seed)
    replace_size = int((replaced_frac * pop_size) // 2 * 2)
    keep_size = pop_size - replace_size
    max_idle_iters = 1

    _iter_ = iter(range(max_iter))
    for iteration in _iter_:
        if sample_X is None:
            # Generate a uniform sample
            sample_X = initialize_uniform(
                feat_dict=feat_dict, size=pop_size, seed=new_seed(rng)
            )
            evals, best_index, median_index = _get_evals(sample_X, eval_func)
        if verbose:
            print(
                "\niteration:",
                iteration,
                "\nbest score:",
                evals[best_index],
                "\nbest sample:",
                ";".join(str(feature_val) for feature_val in sample_X[best_index]),
            )
        # Build new population
        is_new_sample = False
        for idle_iters in range(n_iter_no_change):
            if len(score_per_iter) <= iteration:
                score_per_iter.append(
                    (iteration, evals[best_index], sample_X[best_index])
                )
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
                crossover_func,
                new_seed(rng),
            )
            _mutate_population(new_sample_X, feat_dict, mutation_rate, new_seed(rng))
            # Calculate sample evals
            new_evals, new_best_index, new_median_index = _get_evals(
                new_sample_X, eval_func
            )
            if (new_evals[new_best_index] - evals[best_index]) >= epsilon:
                sample_X, evals, best_index, median_index, is_new_sample = (
                    new_sample_X,
                    new_evals,
                    new_best_index,
                    new_median_index,
                    True,
                )
                score_per_iter.append(
                    (iteration + 1, evals[best_index], sample_X[best_index])
                )
                break
            elif (iteration := next(_iter_, None)) is None:
                warnings.warn(
                    f"Stochastic Optimizer: Maximum iterations ({max_iter}) reached and the optimization hasn't converged yet.",
                    RuntimeWarning,
                )
                break
            if (
                math.isclose(
                    new_evals[new_best_index], evals[best_index], abs_tol=epsilon
                )
                and new_evals[new_median_index] > evals[median_index]
            ):
                sample_X, evals, best_index, median_index = (
                    new_sample_X,
                    new_evals,
                    new_best_index,
                    new_median_index,
                )
        if is_new_sample == False:
            total_fevals = fevals_per_iter * len(score_per_iter)
            last_elements_count = n_iter_no_change - max_idle_iters
            del score_per_iter[-last_elements_count:]
            break
        else:
            max_idle_iters = max(max_idle_iters, idle_iters)
    return sample_X[best_index], evals[best_index], score_per_iter, total_fevals


# Helper functions
def _get_evals(sample_X, eval_func):
    evals = np.asarray([eval_func(sample_x) for sample_x in sample_X])
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
    _evals_ = evals.astype(float) - np.min(evals)  # make minimum zero
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
):
    rng = np.random.default_rng(seed)
    _evals_ = evals.astype(float) - np.min(evals)  # make minimum zero
    _evals_ += 0.01 * np.max(_evals_)  # add minute value
    parents_pairs = rng.choice(
        a=sample_X,
        size=(offspring_size // 2, 2),
        p=_evals_ / np.sum(_evals_),
    )
    parents_pairs_seeds = zip(parents_pairs, new_seed(rng, size=len(parents_pairs)))
    offsprings = np.asarray(
        [
            crossover_func(*parents_pair_seed)
            for parents_pair_seed in parents_pairs_seeds
        ]
    )
    return offsprings.reshape(-1, offsprings.shape[-1])  # collapse one dimension


def _mutate_population(input_X, feat_dict, mutation_rate, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.choice(
        a=len(input_X),
        size=int(mutation_rate * len(input_X)),
        replace=False,
    )
    indices_seeds = zip(indices, new_seed(rng, size=len(indices)))
    input_X[indices] = np.asarray(
        [
            one_variable_uniform(feat_dict, input_X[indx], indx_seed)
            for indx, indx_seed in indices_seeds
        ]
    )
