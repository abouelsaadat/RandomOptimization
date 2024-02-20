"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import numpy as np
from ..utils.sampling import new_seed, initialize_uniform, one_variable_uniform


def optimize(
    feat_dict,
    eval_func,
    n_iter_no_change=None,
    update_no_change=False,
    max_iter=10000,
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
                "-".join(str(feature_val) for feature_val in best_sample),
            )
        new_sample, new_score = _hillclimb(
            feat_dict,
            eval_func,
            initialize_uniform(feat_dict, size=None, seed=new_seed(rng)),
            n_iter_no_change,
            update_no_change,
            _iter_,
            new_seed(rng),
        )
        if new_score > best_score:
            best_sample = new_sample
            best_score = new_score
    return best_sample, best_score


# Helper functions
def _hillclimb(
    feat_dict,
    eval_func,
    input_x,
    n_iter_no_change,
    update_no_change,
    _iter_,
    seed,
):
    rng = np.random.default_rng(seed)
    best_sample, best_score = input_x, eval_func(input_x)
    while True:
        is_new_sample = False
        _jter_ = iter(range(n_iter_no_change))
        while True:
            new_sample = one_variable_uniform(
                feat_dict=feat_dict, sample_x=best_sample, seed=new_seed(rng)
            )
            new_score = eval_func(new_sample)
            if new_score > best_score:
                best_sample, best_score, is_new_sample = new_sample, new_score, True
                break
            elif update_no_change and new_score == best_score:
                best_sample, best_score = new_sample, new_score
            # Increment iterators
            if next(_jter_, None) is None:
                break
            if next(_iter_, None) is None:
                break
        if is_new_sample == False:
            break
        elif next(_iter_, None) is None:
            break
    return best_sample, best_score
