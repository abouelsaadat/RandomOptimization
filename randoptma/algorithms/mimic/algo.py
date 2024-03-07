""" MIMIC Random Optimization Algorithm. """

# Author: Mohamed Abouelsaadat
# License: MIT

import io
import math
import warnings
import contextlib
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from .utils.maxspantree import build_mst
from ..utils.sampling import new_seed


def optimize(
    feat_dict,
    eval_func,
    n_samples=1000,
    top_percentile=0.2,
    weighted=False,
    n_iter_no_change=10,
    max_iter=int(1e10),
    epsilon=1e-3,
    seed=None,
    verbose=False,
):
    """Implementation of MIMIC algorithm according to De Bonet, J., C. Isbell, and P. Viola (1997). MIMIC: Finding Optima by
       Estimating Probability Densities. In *Advances in Neural Information Processing Systems* (NIPS) 9, pp. 424-430.

    Params
    ------
    feat_dict: dictionary with keys representing features indices, and values representing valid values.
        discrete ex : [0,1,2,3,4]
        continuous ex : (-1, 1)
    eval_func: evaluation function used to measure performance of each sample.
    n_samples: positive integer value representing the sample size to be used
    top_percentile: the fraction of samples that ought to be used in constructing the bayesian network
    weighted: Whether to take into account samples' fitness scores while building dependency trees or not
    n_iter_no_change: number of iterations with no change in best score to determine convergence
    max_iter: total max iterations allowed
    epsilon: smallest change taken into account as improvement
    seed: random seed to be used in random numbers generation, if None an arbitrary random seed is chosen
    verbose: boolean value to switch on/off the printing of each iteration results

    Return
    ------
    sample with highest score, highest score, array of iteration number vs score, number of function evaluations per iteration
    """
    sample_X = None
    score_per_iter = list()
    fevals_per_iter = n_samples
    max_idle_iters = 1
    rng = np.random.default_rng(seed)
    _iter_ = iter(range(max_iter))
    for iteration in _iter_:
        if sample_X is None:
            sample_X = _generate_new_samples(
                n_samples=n_samples, feat_dict=feat_dict, seed=new_seed(rng)
            )
            evals, best_index, median_index, top_percentile_indices = _get_evals(
                top_percentile=top_percentile, sample_X=sample_X, eval_func=eval_func
            )
        if verbose:
            print(
                "\niteration:",
                iteration,
                "\nbest score:",
                evals[best_index],
                "\nbest sample:",
                ";".join(str(feature_val) for feature_val in sample_X[best_index]),
            )
        # Build new population
        is_new_sample = False
        new_model_required = True
        for idle_iters in range(n_iter_no_change):
            if len(score_per_iter) <= iteration:
                score_per_iter.append(
                    (iteration, evals[best_index], sample_X[best_index])
                )
            if new_model_required:
                new_model_required = False
                model = _build_model(
                    feat_dict,
                    sample_X,
                    evals,
                    top_percentile_indices,
                    weighted,
                    seed=new_seed(rng),
                )
            new_sample_X = _generate_new_samples(
                n_samples, feat_dict, model, new_seed(rng)
            )
            (
                new_evals,
                new_best_index,
                new_median_index,
                new_top_percentile_indices,
            ) = _get_evals(top_percentile, new_sample_X, eval_func)
            if (new_evals[new_best_index] - evals[best_index]) >= epsilon:
                (
                    sample_X,
                    evals,
                    best_index,
                    median_index,
                    top_percentile_indices,
                    is_new_sample,
                ) = (
                    new_sample_X,
                    new_evals,
                    new_best_index,
                    new_median_index,
                    new_top_percentile_indices,
                    True,
                )
                score_per_iter.append(
                    (iteration + 1, evals[best_index], sample_X[best_index])
                )
                break
            elif (iteration := next(_iter_, None)) is None:
                warnings.warn(
                    f"Stochastic Optimizer: Maximum iterations ({max_iter}) reached and the optimization hasn't converged yet.",
                    RuntimeWarning,
                )
                break
            if (
                math.isclose(
                    new_evals[new_best_index], evals[best_index], abs_tol=epsilon
                )
                and new_evals[new_median_index] > evals[median_index]
            ):
                (
                    sample_X,
                    evals,
                    best_index,
                    median_index,
                    top_percentile_indices,
                    new_model_required,
                ) = (
                    new_sample_X,
                    new_evals,
                    new_best_index,
                    new_median_index,
                    new_top_percentile_indices,
                    True,
                )
        if is_new_sample == False:
            total_fevals = fevals_per_iter * len(score_per_iter)
            last_elements_count = n_iter_no_change - max_idle_iters
            del score_per_iter[-last_elements_count:]
            break
        else:
            max_idle_iters = max(max_idle_iters, idle_iters)
    return sample_X[best_index], evals[best_index], score_per_iter, total_fevals


# Helper Functions
def _generate_new_samples(
    n_samples,
    feat_dict,
    model=None,
    seed=None,
):
    sorted_keys = sorted(feat_dict.keys())
    if model is None:
        model = BayesianNetwork()
        model.add_nodes_from(sorted_keys)
        data = pd.DataFrame(data={key: [] for key in sorted_keys})
        model.fit(data, state_names=feat_dict)
    # Suppress console output and warnings
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        sample_X = model.simulate(n_samples=n_samples, seed=seed, show_progress=False)[
            sorted_keys
        ].to_numpy()
    return sample_X


def _get_evals(top_percentile, sample_X, eval_func):
    evals = np.asarray([eval_func(x) for x in sample_X])
    order = np.argsort(evals)
    best_index = order[len(sample_X) - 1]
    median_index = order[len(sample_X) // 2]
    top_n_samples = int(top_percentile * len(sample_X))
    top_percentile_indices = order[-top_n_samples:]
    return evals, best_index, median_index, top_percentile_indices


def _build_model(
    feat_dict, sample_X, evals, top_percentile_indices, weighted, seed=None
):
    dict_keys = [*feat_dict]
    rng = np.random.default_rng(seed)
    top_evals = evals[top_percentile_indices]
    top_sample_X = sample_X[top_percentile_indices]
    edges = build_mst(rng.choice(dict_keys), dict_keys, top_sample_X)
    model = _fit_bayesian_model(feat_dict, edges, top_sample_X, top_evals, weighted)
    return model


def _fit_bayesian_model(feat_dict, edges, top_sample_X, top_evals, weighted):
    sorted_keys = sorted(feat_dict.keys())
    model = BayesianNetwork(edges)
    data = pd.DataFrame(data={key: top_sample_X[:, key] for key in sorted_keys})
    if weighted:
        data["_weight"] = top_evals - np.min(top_evals)  # make minimum zero
    model.fit(
        data,
        state_names=feat_dict,
        estimator=BayesianEstimator,
        prior_type="K2",
        weighted=weighted,
        n_jobs=1,
    )
    return model
