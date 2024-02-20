"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import randoptma.mimic.algo as mimic_algo
import randoptma.genetic.algo as genetic_algo
import randoptma.simanneal.algo as simanneal_algo


def flipflop(input_x):
    count = 0
    for itr in range(1, len(input_x)):
        if input_x[itr] != input_x[itr - 1]:
            count += 1
    return count


ENTRY_LENGTH = 50
best_sample, best_score = genetic_algo.optimize(
    {feat: [0, 1] for feat in range(ENTRY_LENGTH)},
    lambda input: flipflop(input),
)
print("best score: ", best_score)
print("best sample: ", ";".join(str(int(bit)) for bit in best_sample))
