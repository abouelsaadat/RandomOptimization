"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import math
import random as rnd
from .decay import ArithmeticGeometric


def optimize(
    feat_dict,
    eval_func,
    cool_schedule=ArithmeticGeometric(),
    n_iter_no_change=1000,
    max_iter=1000,
    seed=None,
    verbose=False,
):
    rnd.seed(seed)
    best_sample = None
    _iter_ = iter(range(max_iter))
    for iteration in _iter_:
        if best_sample is None:
            best_sample = _uniform_sample(feat_dict)
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
            new_sample = _get_neighbor(best_sample, feat_dict)
            new_score = eval_func(new_sample)
            if new_score > best_score or rnd.random() < math.exp(
                (new_score - best_score) / cool_schedule.next_T()
            ):
                best_sample, best_score, is_new_sample = new_sample, new_score, True
                break
            elif next(_iter_, None) is None:
                break
        if is_new_sample == False:
            break
    return best_sample, best_score


def _uniform_sample(feat_dict):
    sample = list()
    for key in feat_dict:
        if _is_discrete_format(feat_dict[key]):
            sample.append(rnd.choice(feat_dict[key]))
        elif _is_continuous_format(feat_dict[key]):
            sample.append(
                rnd.uniform(low=min(feat_dict[key]), high=max(feat_dict[key]))
            )
        else:
            raise TypeError(
                "Value of the key <{key}> in features dictionary is wrong, use either tuple for continous features or list for discrete features".format(
                    key=repr(key)
                )
            )
    return sample


def _is_discrete_format(available_values):
    return type(available_values) is list and len(available_values) > 1


def _is_continuous_format(available_values):
    return type(available_values) is tuple and len(available_values) > 1


def _get_neighbor(input_x, feat_dict):
    output_x = input_x.copy()
    feat_indx = rnd.randrange(len(feat_dict))
    if _is_discrete_format(feat_dict[feat_indx]):
        output_x[feat_indx] = rnd.choice(
            [val for val in feat_dict[feat_indx] if val != output_x[feat_indx]]
        )
    elif _is_continuous_format(feat_dict[feat_indx]):
        temp_value = output_x[feat_indx]
        while abs(temp_value - output_x[feat_indx]) < 0.1 * (
            max(feat_dict[feat_indx]) - min(feat_dict[feat_indx])
        ):
            temp_value = rnd.uniform(
                low=min(feat_dict[feat_indx]), high=max(feat_dict[feat_indx])
            )
        output_x[feat_indx] = temp_value
    else:
        raise TypeError(
            "Value of the key <{feat_indx}> in features dictionary is wrong, use either tuple for continous features or list for discrete features".format(
                feat_indx=repr(feat_indx)
            )
        )
    return output_x
