"""  """

# Author: Mohamed Abouelsaadat
# License: MIT

import numpy as np
import matplotlib.pyplot as plt
import randoptma.mimic.algo as mimic_algo
import randoptma.genetic.algo as genetic_algo
import randoptma.simanneal.algo as simanneal_algo
import randoptma.randhillclimb.algo as randhillclimb_algo


def tail(b, input_x):
    # Number of trailing bs
    count = 0
    for bit in input_x[::-1]:
        if bit == b:
            count += 1
        else:
            break
    return count


def head(b, input_x):
    # Number of leading bs
    count = 0
    for bit in input_x:
        if bit == b:
            count += 1
        else:
            break
    return count


# Four Peaks Cost Function (Baluja and Caruana, 1995)
def fourpeaks(input_x, T):
    # Calculate R(input, T)
    R_count = 0
    if tail(0, input_x) > T and head(1, input_x) > T:
        R_count = len(input_x)
    else:
        R_count = 0

    return max(tail(0, input_x), head(1, input_x)) + R_count


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
    x_values = ["-".join(str(int(bit)) for bit in value) for value in x_values]

    plt.ylabel("cost function")
    plt.plot(x_values, y_values)
    plt.xticks(x_values[::3], rotation="horizontal")
    plt.show()


ENTRY_LENGTH = 50
ENTRY_T_PEAK = int(0.3 * ENTRY_LENGTH)
best_sample, best_score = randhillclimb_algo.optimize(
    {feat: [0, 1] for feat in range(ENTRY_LENGTH)},
    lambda input: fourpeaks(input, ENTRY_T_PEAK), verbose=True,
)
print("best score: ", best_score)
print("best sample: ", "-".join(str(int(bit)) for bit in best_sample))
