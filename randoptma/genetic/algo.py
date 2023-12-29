"""Genetic Algorithm Optimization based on Tom Mitchell's book: Machine Learning, Chapter 9"""

# Author: Mohamed Abouelsaadat
# License: MIT

import numpy as np
from .crossover import singlepoint


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
            sample_X = _uniform_sample(pop_size, feat_dict, _new_seed(rng))
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


def _uniform_sample(pop_size, feat_dict, seed=None):
    rng = np.random.default_rng(seed)
    sample_X = np.empty([pop_size, len(feat_dict)])
    for key in feat_dict:
        if _is_discrete_format(feat_dict[key]):
            sample_X[:, key] = rng.choice(a=feat_dict[key], size=pop_size)
        elif _is_continuous_format(feat_dict[key]):
            sample_X[:, key] = rng.uniform(
                low=min(feat_dict[key]), high=max(feat_dict[key], size=pop_size)
            )
        else:
            raise TypeError(
                "Value of the key <{key}> in features dictionary is wrong, use either tuple for continous features or list for discrete features".format(
                    key=repr(key)
                )
            )
    return sample_X


def _is_discrete_format(available_values):
    return type(available_values) is list and len(available_values) > 1


def _is_continuous_format(available_values):
    return type(available_values) is tuple and len(available_values) > 1


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
        input_X[sample_indx] = _mutate_sample(
            input_X[sample_indx], feat_dict, _new_seed(rng)
        )


def _mutate_sample(input_x, feat_dict, seed=None):
    rng = np.random.default_rng(seed)
    output_x = input_x.copy()
    feat_indx = rng.integers(len(feat_dict))
    if _is_discrete_format(feat_dict[feat_indx]):
        output_x[feat_indx] = _mutate_discrete(
            output_x[feat_indx], feat_dict[feat_indx], _new_seed(rng)
        )
    elif _is_continuous_format(feat_dict[feat_indx]):
        output_x[feat_indx] = _mutate_continuous(
            output_x[feat_indx], feat_dict[feat_indx], _new_seed(rng)
        )
    else:
        raise TypeError(
            "Value of the key <{feat_indx}> in features dictionary is wrong, use either tuple for continous features or list for discrete features".format(
                feat_indx=repr(feat_indx)
            )
        )
    return output_x


def _mutate_discrete(value, available_values, seed=None):
    rng = np.random.default_rng(seed)
    value_index = available_values.index(value)
    weights = [-1 * abs(value_index - itr) for itr in range(len(available_values))]
    max_diff = -1 * min(weights) + 1
    weights = np.asarray([weight + max_diff for weight in weights])
    weights[value_index] = 0
    return rng.choice(a=available_values, p=weights / np.sum(weights))


def _mutate_continuous(value, available_values, seed=None):
    rng = np.random.default_rng(seed)
    temp_value = value
    min_value = max(available_values)
    max_value = min(available_values)
    while abs(temp_value - value) < 0.1 * (max_value - min_value):
        temp_value = rng.triangular(
            min_value,
            value,
            max_value,
        )
    return temp_value
