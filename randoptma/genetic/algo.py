"""Genetic Algorithm Optimization based on Tom Mitchell's book: Machine Learning, Chapter 9"""

# Author: Mohamed Abouelsaadat
# License: MIT

import numpy as np
from .crossover import singlepoint
from ..utils.sampling import initialize_uniform, one_variable_triangular_rounded


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
):
    sample_X = None
    rng = np.random.default_rng(seed)
    base_sample_size = int(((1 - replaced_frac) * pop_size) // 2 * 2)

    _iter_ = iter(range(max_iter))
    for iteration in _iter_:
        if sample_X is None:
            # Generate a uniform sample
            sample_X = initialize_uniform(
                feat_dict=feat_dict, size=pop_size, seed=_new_seed(rng)
            )
            evals, best_index, median_index = _get_evals(sample_X, eval_func)
        if verbose:
            print(
                "\niteration:",
                iteration,
                "\nbest score:",
                evals[best_index],
                "\nbest sample:",
                "-".join(str(int(bit)) for bit in sample_X[best_index]),
            )
        # Build new population
        is_new_sample = False
        for _ in range(n_iter_no_change):
            new_sample_X = np.empty([pop_size, len(feat_dict)])
            new_sample_X[:base_sample_size, :] = _next_population_base(
                sample_X,
                evals,
                base_sample_size,
                _new_seed(rng),
            )
            new_sample_X[base_sample_size:, :] = _produce_offsprings(
                sample_X,
                evals,
                int(pop_size - base_sample_size),
                lambda parents: crossover_func(parents, _new_seed(rng)),
                _new_seed(rng),
            )
            _mutate_population(new_sample_X, feat_dict, mutation_rate, _new_seed(rng))
            # Calculate sample evals
            new_evals, new_best_index, new_median_index = _get_evals(
                new_sample_X, eval_func
            )
            if (
                new_evals[new_best_index] > evals[best_index]
                and new_evals[new_median_index] >= evals[median_index]
                or new_evals[new_best_index] >= evals[best_index]
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
                break
        if is_new_sample == False:
            break
    return sample_X[best_index], evals[best_index]


# Helper functions
def _new_seed(rng):
    return rng.integers(1e5)


def _get_evals(sample_X, eval_func):
    evals = np.asarray([eval_func(x) for x in sample_X])
    order = np.argsort(evals)
    best_index = order[len(sample_X) - 1]
    median_index = order[len(sample_X) // 2]
    return evals, best_index, median_index


def _next_population_base(
    sample_X,
    evals,
    base_sample_size,
    seed=None,
):
    rng = np.random.default_rng(seed)
    population_base = rng.choice(
        a=sample_X,
        size=base_sample_size,
        p=evals / np.sum(evals),
    )
    return population_base


def _produce_offsprings(
    sample_X,
    evals,
    offspring_size,
    crossover_func,
    seed=None,
):
    rng = np.random.default_rng(seed)
    parents_pairs = rng.choice(
        a=sample_X,
        size=(offspring_size // 2, 2),
        p=evals / np.sum(evals),
    )
    offsprings = np.empty([offspring_size, len(sample_X[0])])
    for parents_pair_indx in range(len(parents_pairs)):
        indx = 2 * parents_pair_indx
        offsprings[indx : indx + 2, :] = crossover_func(
            parents_pairs[parents_pair_indx]
        )
    return offsprings


def _mutate_population(input_X, feat_dict, mutation_rate, seed=None):
    rng = np.random.default_rng(seed)
    for sample_indx in rng.choice(
        a=len(input_X),
        size=int(mutation_rate * len(input_X)),
        replace=False,
    ):
        input_X[sample_indx] = one_variable_triangular_rounded(
            feat_dict=feat_dict,
            sample_x=input_X[sample_indx],
            seed=_new_seed(rng),
        )
