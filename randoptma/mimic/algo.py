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
    n_iter_no_change=10,
    max_iter=1000,
    seed=None,
    verbose=False,
):
    sample_X = None
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
                "-".join(str(int(bit)) for bit in sample_X[best_index]),
            )
        dict_keys = [*feat_dict]
        top_evals = evals[top_percentile_indices]
        top_sample_X = sample_X[top_percentile_indices]
        edges = build_mst(rng.choice(dict_keys), dict_keys, top_sample_X)
        model = _fit_bayesian_model(feat_dict, edges, top_sample_X, top_evals)
        is_new_sample = False
        for _ in range(n_iter_no_change):
            new_sample_X = _generate_new_samples(
                n_samples, feat_dict, model, new_seed(rng)
            )
            (
                new_evals,
                new_best_index,
                new_median_index,
                new_top_percentile_indices,
            ) = _get_evals(top_percentile, new_sample_X, eval_func)
            if (
                new_evals[new_best_index] > evals[best_index]
                or math.isclose(new_evals[new_best_index], evals[best_index])
                and new_evals[new_median_index] > evals[median_index]
            ):
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
                break
            elif next(_iter_, None) is None:
                warnings.warn(
                    f"Stochastic Optimizer: Maximum iterations ({max_iter}) reached and the optimization hasn't converged yet.",
                    RuntimeWarning,
                )
                break
        if is_new_sample == False:
            break
    return sample_X[best_index], evals[best_index]


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
        sample_X = model.simulate(n_samples=n_samples, seed=seed)[
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


def _fit_bayesian_model(feat_dict, edges, top_sample_X, top_evals):
    sorted_keys = sorted(feat_dict.keys())
    model = BayesianNetwork(edges)
    data = pd.DataFrame(data={key: top_sample_X[:, key] for key in sorted_keys})
    data["_weight"] = top_evals - np.min(top_evals)  # make minimum zero
    model.fit(
        data,
        state_names=feat_dict,
        estimator=BayesianEstimator,
        prior_type="K2",
        weighted=True,
    )
    return model
