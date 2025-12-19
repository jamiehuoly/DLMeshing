import torch
import numpy as np
from torch_geometric.utils import to_undirected, coalesce


def get_graph_edges(coordinate, elems_numpy):
    # tetrahedral edges combinations
    all_edges = []
    combinations = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
    ]
    # iterate combinations to extract tags for each edge
    for i, j in combinations:
        edge_pair = elems_numpy[:, [i, j]]
        all_edges.append(edge_pair)

    edges_stacked = np.vstack(all_edges)
    edges_stacked_transpose = edges_stacked.transpose()
    edge_index = torch.tensor(edges_stacked_transpose, dtype=torch.long)
    print(edge_index)
    print(edge_index.shape)
    # undirected and eliminate repeated edges
    edge_index = to_undirected(edge_index)
    print(edge_index)
    print(edge_index.shape)
    edge_index = coalesce(edge_index)

    return edge_index