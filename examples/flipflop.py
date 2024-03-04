"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import randoptma.algorithms.mimic.algo as mimic_algo
import randoptma.algorithms.genetic.algo as genetic_algo
import randoptma.algorithms.simanneal.algo as simanneal_algo


def flipflop(input_x):
    count = 0
    for itr in range(1, len(input_x)):
        if input_x[itr] != input_x[itr - 1]:
            count += 1
    return count


ENTRY_LENGTH = 50


def problem_eval_function(input):
    return flipflop(input)


def problem_feat_dict():
    return {feat: [0, 1] for feat in range(ENTRY_LENGTH)}


best_sample, best_score, *_ = genetic_algo.optimize(
    problem_feat_dict(),
    problem_eval_function,
)
print("best score: ", best_score)
print("best sample: ", ";".join(str(int(bit)) for bit in best_sample))
