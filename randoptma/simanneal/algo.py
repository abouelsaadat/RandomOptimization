"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import math
import numpy as np
from .decay import ArithmeticGeometric
from ..utils.sampling import new_seed, initialize_uniform, one_variable_uniform


def optimize(
    feat_dict,
    eval_func,
    cool_schedule=ArithmeticGeometric(),
    n_iter_no_change=None,
    max_iter=1000,
    seed=None,
    verbose=False,
):
    rng = np.random.default_rng(seed)
    best_sample = None
    n_iter_no_change = (
        int(1.5 * len(feat_dict)) if n_iter_no_change is None else n_iter_no_change
    )
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
                "-".join(str(int(bit)) for bit in best_sample),
            )
        is_new_sample = False
        for _ in range(n_iter_no_change):
            new_sample = one_variable_uniform(
                feat_dict=feat_dict, sample_x=best_sample, seed=new_seed(rng)
            )
            new_score = eval_func(new_sample)
            if new_score > best_score or rng.random() < math.exp(
                (new_score - best_score) / cool_schedule.next_T()
            ):
                best_sample, best_score, is_new_sample = new_sample, new_score, True
                break
            elif next(_iter_, None) is None:
                break
        if is_new_sample == False:
            break
    return best_sample, best_score
