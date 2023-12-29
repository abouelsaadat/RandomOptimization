""" Implementation of a max spanning tree based on mutual information using prim's algorithm """

# Author: Mohamed Abouelsaadat
# License: MIT

from .minheap import MinHeapQueue
from .information import mutual_information


def build_mst(start_node, node_names, sample_X):
    edges = list()
    chosen_node = start_node
    processed_nodes = set()
    edge_heap = MinHeapQueue()
    unprocessed_nodes = set(node_names)
    while len(unprocessed_nodes) > 0:
        processed_nodes.add(chosen_node)
        unprocessed_nodes.remove(chosen_node)
        for node in unprocessed_nodes:
            edge_heap.push(
                (
                    -1.0
                    * mutual_information(
                        sample_X[:, chosen_node],
                        sample_X[:, node],
                    ),
                    (chosen_node, node),
                )
            )
        while len(unprocessed_nodes) > 0 and len(edge_heap) > 0:
            edge = edge_heap.pop()[1]
            if edge[1] not in processed_nodes:
                edges.append(edge)
                chosen_node = edge[1]
                break
    return edges
