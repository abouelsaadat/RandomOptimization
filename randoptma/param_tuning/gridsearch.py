""" """

# Author: Mohamed Abouelsaadat
# License: MIT

import time
import itertools
import numpy as np
from multiprocess import Pool, Lock

print_mutex = Lock()


def gridsearch(
    optimizer_func,
    feat_dict,
    eval_func,
    optimizer_params=None,
    seed=None,
    n_runs=10,
    n_jobs=1,
    verbose=False,
):
    if optimizer_params is None:
        permutations_params = [{}]
    else:
        keys, values = zip(*optimizer_params.items())
        values = [val if hasattr(val, "__iter__") else (val,) for val in values]
        permutations_params = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_size = n_runs * len(permutations_params)
    seeds = np.random.default_rng(seed).integers(1e5, size=total_size)
    seeded_opt_problem = [
        {"feat_dict": feat_dict, "eval_func": eval_func, "seed": seed} for seed in seeds
    ]
    executer_params = zip(
        total_size * (optimizer_func,),
        n_runs * permutations_params,
        seeded_opt_problem,
        total_size * (verbose,),
        range(total_size),
        total_size * (total_size,),
    )
    with Pool(n_jobs) as p:
        scores = p.starmap(_executer, executer_params)
    result = list()
    for itr in range(len(permutations_params)):
        result.append(permutations_params[itr].copy())
        result[itr]["scores"] = list()
        result[itr]["times"] = list()
        result[itr]["iterations"] = list()
        result[itr]["fevals"] = list()
    for itr in range(len(scores)):
        result[itr % len(permutations_params)]["scores"].append(scores[itr][0])
        result[itr % len(permutations_params)]["times"].append(scores[itr][1])
        result[itr % len(permutations_params)]["iterations"].append(scores[itr][2])
        result[itr % len(permutations_params)]["fevals"].append(scores[itr][3])
    return result


def _executer(
    optimizer_func,
    permutations_params,
    seeded_opt_problem,
    verbose,
    input_index,
    input_size,
):
    if verbose:
        with print_mutex:
            print(f"{input_index + 1}/{input_size} Starting:", permutations_params)
    start = time.time()
    _, best_score, score_per_iter, total_fevals = optimizer_func(
        **seeded_opt_problem, **permutations_params
    )
    end = time.time()
    if verbose:
        with print_mutex:
            print(
                f"{input_index + 1}/{input_size}",
                f"Ended: {permutations_params};",
                f"Score: {best_score};",
                f"Time: {_float_format(end - start)} sec",
            )
    return (
        best_score,
        (end - start),
        len(score_per_iter),
        total_fevals,
    )


def _float_format(val):
    return f"{val:1.0e}" if val < 1e-2 else f"{val:1.2f}"
