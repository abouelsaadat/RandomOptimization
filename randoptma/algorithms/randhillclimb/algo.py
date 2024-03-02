"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import math
import warnings
import numpy as np
from ..utils.sampling import new_seed, initialize_uniform, one_variable_uniform


def optimize(
    feat_dict,
    eval_func,
    n_iter_no_change=100,
    n_restarts_no_change=2,
    update_no_change=False,
    max_iter=int(1e10),
    max_restarts=int(1e10),
    epsilon=1e-3,
    seed=None,
    verbose=False,
):
    """Implementation of Random Restart Hill Climbing Optimization Algorithm.
    It starts as a normal Hill Climbing algorithm then restart from a new sample on convergence,
    It keeps iterating till it reaches max number of allowed iterations.
    To get the behavior of a normal Hill Climbing without any restart set `max_restarts` to zero.

    Params
    ------
    feat_dict: dictionary with keys representing features indices, and values representing valid values.
        discrete ex : [0,1,2,3,4]
        continuous ex : (-1, 1)
    eval_func: evaluation function used to measure performance of each sample.
    n_iter_no_change: number of iterations with no change in best score to
                      determine convergence for single hill climbing searches
    n_restarts_no_change: number of restarts after which the search should end
                          if no improvement was captured
    update_no_change: whether to update to a newer sample with same score or
                      not while testing for convergence, This could help traverse
                      the plateau if the search is stuck in one.
    max_iter: total max iterations allowed
    max_restarts: max number of restarts allowed, value of 0 makes it
                  equivalent to single run of hill climbing
    epsilon: smallest change taken into account as improvement
    seed: random seed to be used in random numbers generation, if None an arbitrary
          random value would be used as a seed.
    verbose: boolean value to switch on/off the printing of each iteration results

    Return
    ------
    sample with highest score, highest score, array of iteration number vs score, number of function evaluations per iteration
    """
    rng = np.random.default_rng(seed)
    restart = False
    n_restarts = 0
    restarts_since_update = 0
    best_sample = None
    score_per_iter = list()
    fevals_per_iter = 1
    max_idle_iters = 1
    end_pos = []
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
        for idle_iters in range(n_iter_no_change):
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
            if (new_score - local_best_score) >= epsilon or restart == True:
                local_best_sample, local_best_score, is_new_sample = (
                    new_sample,
                    new_score,
                    True,
                )
                if restart:
                    restart = False
                    n_restarts += 1
                    restarts_since_update += 1
                break
            elif (iteration := next(_iter_, None)) is None:
                warnings.warn(
                    f"Stochastic Optimizer: Maximum iterations ({max_iter}) reached and the optimization hasn't converged yet.",
                    RuntimeWarning,
                )
                break
            if update_no_change and math.isclose(
                new_score, local_best_score, abs_tol=epsilon
            ):
                local_best_sample, local_best_score = new_sample, new_score
        if local_best_score > best_score:
            restarts_since_update = 0
            best_sample, best_score = local_best_sample, local_best_score
            score_per_iter.append((iteration + 1, best_score))
        if is_new_sample == False:
            end_pos.append(len(score_per_iter))
            if (
                restarts_since_update < n_restarts_no_change
                and n_restarts < max_restarts
            ):
                restart = True
            else:
                last_elements_count = n_iter_no_change - max_idle_iters
                # Remove extra iterations in each single run
                for indx in end_pos[::-1]:
                    del score_per_iter[indx - last_elements_count : indx]
                # Correct for iterations numbers
                for indx in range(len(score_per_iter)):
                    score_per_iter[indx] = (indx, score_per_iter[indx][1])
                break
        else:
            max_idle_iters = max(max_idle_iters, idle_iters)
    return best_sample, best_score, score_per_iter, fevals_per_iter
