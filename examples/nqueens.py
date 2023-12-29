"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import randoptma.mimic.algo as mimic_algo
import randoptma.genetic.algo as genetic_algo
import randoptma.simanneal.algo as simanneal_algo
import randoptma.randhillclimb.algo as randhillclimb_algo


def nqueens(input_x):
    queens = set(range(len(input_x)))
    for itr in range(len(input_x)):
        for jtr in range(itr + 1, len(input_x)):
            if input_x[itr] == input_x[jtr] or (jtr - itr) == abs(
                input_x[jtr] - input_x[itr]
            ):
                queens.discard(itr)
                queens.discard(jtr)
    return len(queens)


ENTRY_LENGTH = 50
best_sample, best_score = simanneal_algo.optimize(
    {feat: list(range(ENTRY_LENGTH)) for feat in range(ENTRY_LENGTH)},
    lambda input: nqueens(input),
)
print("best score: ", best_score)
print("best sample: ", "-".join(str(int(pos)) for pos in best_sample))
