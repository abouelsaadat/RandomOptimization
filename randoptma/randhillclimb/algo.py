"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import random as rnd


def optimize(
    feat_dict,
    eval_func,
    n_iter_no_change=None,
    update_no_change=False,
    max_iter=10000,
    seed=None,
    verbose=False,
):
    rnd.seed(seed)
    best_sample = None
    n_iter_no_change = len(feat_dict) if n_iter_no_change is None else n_iter_no_change
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
        new_sample, new_score = _hillclimb(
            feat_dict,
            eval_func,
            _uniform_sample(feat_dict),
            n_iter_no_change,
            update_no_change,
            _iter_,
        )
        if new_score > best_score:
            best_sample = new_sample
            best_score = new_score
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


def _hillclimb(
    feat_dict, eval_func, input_x, n_iter_no_change, update_no_change, _iter_
):
    best_sample, best_score = input_x, eval_func(input_x)
    while True:
        is_new_sample = False
        _jter_ = iter(range(n_iter_no_change))
        while True:
            new_sample = _get_neighbor(best_sample, feat_dict)
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
