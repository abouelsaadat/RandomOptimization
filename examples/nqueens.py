"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import time
import randoptma.algorithms.mimic.algo as mimic_algo
import randoptma.algorithms.genetic.algo as genetic_algo
import randoptma.algorithms.simanneal.algo as simanneal_algo
import randoptma.algorithms.randhillclimb.algo as randhillclimb_algo


def nqueens(input_x):
    queens = set(range(len(input_x)))
    for itr in range(len(input_x)):
        for jtr in range(itr + 1, len(input_x)):
            if input_x[itr] == input_x[jtr] or (jtr - itr) == abs(
                input_x[jtr] - input_x[itr]
            ):
                queens.discard(jtr)
    return len(queens)


ENTRY_LENGTH = 50


def problem_eval_function(input):
    return nqueens(input)


def problem_feat_dict():
    return {feat: list(range(ENTRY_LENGTH)) for feat in range(ENTRY_LENGTH)}


start = time.time()
best_sample, best_score, *_ = simanneal_algo.optimize(
    problem_feat_dict(),
    problem_eval_function,
)
end = time.time()
print("elapsed time:", end - start)
print("best score: ", best_score)
print("best sample: ", ";".join(str(int(pos)) for pos in best_sample))
