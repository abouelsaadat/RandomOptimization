""" """

# Author: Mohamed Abouelsaadat
# License: MIT

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import randoptma.algorithms.mimic.algo as mimic_algo
import randoptma.algorithms.genetic.algo as genetic_algo
import randoptma.algorithms.simanneal.algo as simanneal_algo
import randoptma.algorithms.randhillclimb.algo as randhillclimb_algo


def kcolor(x_sample, edges):
    score = len(x_sample)
    for edge in edges:
        if (
            edge[0] == edge[1]
            or edge[0] > edge[1]
            and [edge[1], edge[0]] in edges.tolist()
        ):
            continue
        if x_sample[edge[0]] == x_sample[edge[1]]:
            score -= 1
    return score


K = 3
ENTRY_LENGTH = 10
rng = np.random.default_rng()
edges = rng.integers(ENTRY_LENGTH, size=(rng.integers(2 * K, ENTRY_LENGTH**2 / 4), 2))
edges = np.delete(
    edges, np.where(edges[:, 0] == edges[:, 1])[0], axis=0
)  # delete circular edges
best_sample, best_score, _, _ = mimic_algo.optimize(
    {feat: list(range(K)) for feat in range(ENTRY_LENGTH)},
    lambda input: kcolor(input, edges),
)
print("best score: ", best_score)
print("best sample: ", ";".join(str(int(bit)) for bit in best_sample))

G = nx.Graph()
G.add_nodes_from(range(ENTRY_LENGTH))
G.add_edges_from(edges)
nx.draw_networkx(G, node_color=["C" + str(val) for val in best_sample])
plt.show()
