""" """

# Author: Mohamed Abouelsaadat
# License: MIT

import time
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
import randoptma.mimic.algo as mimic_algo
import randoptma.genetic.algo as genetic_algo
import randoptma.simanneal.algo as simanneal_algo
import randoptma.randhillclimb.algo as randhillclimb_algo


def calculate_length_layers(layers):
    total_length = 0
    for itr in range(1, len(layers)):
        total_length += layers[itr - 1] * layers[itr] + layers[itr]
    return total_length


def pack_weights(flattened_array, layers):
    """pack a flattened array into mlp clf coefs and intecepts"""
    indx = 0
    coefs = list()
    intercepts = list()
    for itr in range(1, len(layers)):
        # Fill in weights matrix
        length = layers[itr - 1] * layers[itr]
        coefs.append(
            flattened_array[indx : indx + length].reshape(
                (layers[itr - 1], layers[itr])
            )
        )
        indx += length
        # Fill in bias vector
        intercepts.append(flattened_array[indx : indx + layers[itr]])
        indx += layers[itr]
    return coefs, intercepts


def evaluate_mlp_clf(mlp_clf, coefs, intercepts, input_X, input_y):
    mlp_clf.coefs_, mlp_clf.intercepts_ = coefs, intercepts
    return mlp_clf.score(input_X, input_y)


X, y = make_classification(n_samples=100, random_state=1)
clf = MLPClassifier(random_state=1, max_iter=300).fit(X, y)
layers = (clf.n_features_in_, *clf.get_params()["hidden_layer_sizes"], clf.n_outputs_)
print("default score:", clf.score(X, y))

start = time.time()
ENTRY_LENGTH = calculate_length_layers(layers)
best_sample, best_score = genetic_algo.optimize(
    {feat: (-1, 1) for feat in range(ENTRY_LENGTH)},
    lambda input: evaluate_mlp_clf(clf, *pack_weights(input, layers), X, y),
)
end = time.time()
print("elapsed time:", end - start)
print("RO score:", best_score)
