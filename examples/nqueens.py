"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import time
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


start = time.time()
ENTRY_LENGTH = 50
best_sample, best_score = genetic_algo.optimize(
    {feat: list(range(ENTRY_LENGTH)) for feat in range(ENTRY_LENGTH)},
    lambda input: nqueens(input),
    n_jobs=10,
    seed=0,
)
end = time.time()
print(end - start)
print("best score: ", best_score)
print("best sample: ", ";".join(str(int(pos)) for pos in best_sample))
