"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import warnings
import numpy as np
from ..utils.sampling import new_seed, initialize_uniform, one_variable_uniform


def optimize(
    feat_dict,
    eval_func,
    n_iter_no_change=None,
    update_no_change=False,
    max_iter=10000,
    max_no_restarts=100,
    seed=None,
    verbose=False,
):
    rng = np.random.default_rng(seed)
    restart = False
    no_restarts = 0
    best_sample = None
    score_per_iter = list()
    fevals_per_iter = 1
    n_iter_no_change = (
        int(1.5 * len(feat_dict)) if n_iter_no_change is None else n_iter_no_change
    )
    _iter_ = iter(range(max_iter))
    for iteration in _iter_:
        if best_sample is None:
            local_best_sample = initialize_uniform(
                feat_dict, size=None, seed=new_seed(rng)
            )
            best_sample = local_best_sample.copy()
            local_best_score = eval_func(local_best_sample)
            best_score = local_best_score
        if verbose:
            print(
                "\niteration:",
                iteration,
                "\nbest score:",
                best_score,
                "\nbest sample:",
                ";".join(str(feature_val) for feature_val in best_sample),
            )
        for _ in range(n_iter_no_change):
            if len(score_per_iter) <= iteration:
                score_per_iter.append((iteration, best_score))
            is_new_sample = False
            if restart:
                new_sample = initialize_uniform(
                    feat_dict, size=None, seed=new_seed(rng)
                )
            else:
                new_sample = one_variable_uniform(
                    feat_dict=feat_dict, sample_x=local_best_sample, seed=new_seed(rng)
                )
            new_score = eval_func(new_sample)
            if new_score > local_best_score or restart == True:
                local_best_sample, local_best_score, is_new_sample = (
                    new_sample,
                    new_score,
                    True,
                )
                if restart:
                    restart = False
                    no_restarts += 1
                break
            elif (iteration := next(_iter_, None)) is None:
                warnings.warn(
                    f"Stochastic Optimizer: Maximum iterations ({max_iter}) reached and the optimization hasn't converged yet.",
                    RuntimeWarning,
                )
                break
            if update_no_change and new_score == best_score:
                best_sample, best_score = new_sample, new_score
        if best_score < local_best_score:
            best_sample, best_score = local_best_sample, local_best_score
            score_per_iter.append((iteration + 1, best_score))
        if is_new_sample == False:
            if no_restarts < max_no_restarts:
                restart = True
            else:
                break
    return best_sample, best_score, score_per_iter, fevals_per_iter
