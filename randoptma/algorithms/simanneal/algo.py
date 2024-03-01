"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import math
import warnings
import numpy as np
from .decay import ArithmeticGeometric
from ..utils.sampling import new_seed, initialize_uniform, one_variable_uniform


def optimize(
    feat_dict,
    eval_func,
    cool_schedule=ArithmeticGeometric(),
    n_iter_no_change=1000,
    max_iter=int(1e10),
    seed=None,
    verbose=False,
):
    """Implementation of Simulated Annealing optimization algorithm.

    Params
    ------
    feat_dict: dictionary with keys representing features indices, and values representing valid values.
        discrete ex : [0,1,2,3,4]
        continuous ex : (-1, 1)
    eval_func: evaluation function used to measure performance of each sample.
    cool_schedule: temprature cooling schedule to be used.
    update_no_change: whether to update to a newer sample with same score or not while testing for convergence,
                      This could help traverse the plateau if stuck in one.
    max_iter: total max iterations allowed
    max_no_restarts: max number of iterations allowed
    seed: random seed to be used in random numbers generation, if None an arbitrary random seed is chosen
    verbose: boolean value to switch on/off the printing of each iteration results

    Return
    ------
    sample with highest score, highest score, array of iteration number vs score, number of function evaluations per iteration
    """
    rng = np.random.default_rng(seed)
    best_sample = None
    score_per_iter = list()
    fevals_per_iter = 1
    max_idle_iters = 1
    _iter_ = iter(range(max_iter))
    for iteration in _iter_:
        if best_sample is None:
            best_sample = initialize_uniform(feat_dict, size=None, seed=new_seed(rng))
            best_score = eval_func(best_sample)
        if verbose:
            print(
                "\niteration:",
                iteration,
                "\nbest score:",
                best_score,
                "\nbest sample:",
                ";".join(str(feature_val) for feature_val in best_sample),
            )
        is_new_sample = False
        for idle_iters in range(n_iter_no_change):
            if len(score_per_iter) <= iteration:
                score_per_iter.append((iteration, best_score))
            new_sample = one_variable_uniform(
                feat_dict=feat_dict, sample_x=best_sample, seed=new_seed(rng)
            )
            new_score = eval_func(new_sample)
            if new_score > best_score:
                best_sample, best_score, is_new_sample = new_sample, new_score, True
                score_per_iter.append((iteration + 1, best_score))
                break
            elif (iteration := next(_iter_, None)) is None:
                warnings.warn(
                    f"Stochastic Optimizer: Maximum iterations ({max_iter}) reached and the optimization hasn't converged yet.",
                    RuntimeWarning,
                )
                break
            temp = cool_schedule.next_T()
            if (
                math.isclose(new_score, best_score)
                or temp > 0.0
                and rng.random() < math.exp((new_score - best_score) / temp)
            ):
                best_sample, best_score = new_sample, new_score
        if is_new_sample == False:
            last_elements_count = n_iter_no_change - max_idle_iters
            del score_per_iter[-last_elements_count:]
            break
        else:
            max_idle_iters = max(max_idle_iters, idle_iters)
    return best_sample, best_score, score_per_iter, fevals_per_iter
