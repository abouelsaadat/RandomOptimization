"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import numpy as np
import matplotlib.pyplot as plt
import randoptma.algorithms.mimic.algo as mimic_algo
import randoptma.algorithms.genetic.algo as genetic_algo
import randoptma.algorithms.simanneal.algo as simanneal_algo
import randoptma.algorithms.randhillclimb.algo as randhillclimb_algo


def max_run(b, input_x):
    result = 0
    run_count = 0
    for bit in input_x:
        if bit == b:
            run_count += 1
            if run_count > result:
                result = run_count
        else:
            run_count = 0

    return result


def contpeaks(input_x, T):
    # Calculate R(input, T)
    R_count = 0
    if max_run(0, input_x) > T and max_run(1, input_x) > T:
        R_count = len(input_x)
    else:
        R_count = 0

    return max(max_run(0, input_x), max_run(1, input_x)) + R_count


def bitfield(n, size):
    array = np.array([1 if digit == "1" else 0 for digit in bin(n)[2:]])
    return np.concatenate((np.zeros(size - array.size), array))


# Visualization
if False:
    sample_size = 10
    sample_t_peak = int(0.3 * sample_size)

    x_values = list([0])
    for _ in range(sample_size):
        x_values.append(x_values[-1] * 2 + 1)
    for _ in range(sample_size - 1):
        x_values.append(x_values[-1] * 2 % 2**sample_size)

    x_values = [bitfield(n, sample_size) for n in x_values]
    y_values = np.array([sixpeaks(value, sample_t_peak) for value in x_values])
    x_values = [";".join(str(int(bit)) for bit in value) for value in x_values]

    plt.ylabel("cost function")
    plt.plot(x_values, y_values)
    plt.xticks(x_values[::3], rotation="horizontal")
    plt.show()


ENTRY_LENGTH = 50
ENTRY_T_PEAK = int(0.3 * ENTRY_LENGTH)
best_sample, best_score, _, _ = mimic_algo.optimize(
    {feat: [0, 1] for feat in range(ENTRY_LENGTH)},
    lambda input: contpeaks(input, ENTRY_T_PEAK),
)
print("best score: ", best_score)
print("best sample: ", ";".join(str(int(bit)) for bit in best_sample))
