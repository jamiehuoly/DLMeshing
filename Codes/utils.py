import gmsh
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce

def create_tg_data(x, edge_index):
    coordinate_tensor = torch.tensor(x, dtype=torch.float)
    graph_data = Data(x=coordinate_tensor, edge_index=edge_index)
    return graph_data

# from gmsh-stype to numpy-style
def gmsh_tag_transform():
    # node_tag: node ID list
    # coordinate: [x1, y1, z1, x2, y2, z2, ...] flattened list
    node_tag, coordinate, _ = gmsh.model.mesh.getNodes()

    # reshape to 3-dimension
    coordinate = np.array(coordinate, dtype=np.float32).reshape(len(node_tag), 3)

    # Mirror relation： Gmsh tag -> Python index; Gmsh tag may not continuous
    node_index = {tag: i for i, tag in enumerate(node_tag)}

    element_types, element_tags, node_tags_per_element = gmsh.model.mesh.getElements(dim=3)
    # print(gmsh.model.mesh.getElements(dim=3))
    # print(type(elementTypes))
    # print(elementTags)
    # print(nodeTagsPerEl)

    # 4 means tetrahedral (4 nodes)
    if 4 not in element_types:
        raise ValueError("No tetrahedron found! Please check if 3D mesh was generated.")

    tetra_index = element_types.tolist().index(4)
    tetra_nodes_tags = np.array(node_tags_per_element[tetra_index], dtype=np.int64).reshape(-1, 4)
    print(tetra_nodes_tags)

    # mirroring gmsh coordinate into pytorch coordinate
    elems_numpy = np.zeros_like(tetra_nodes_tags)
    for i in range(tetra_nodes_tags.shape[0]):
        for j in range(tetra_nodes_tags.shape[1]):
            gmsh_tag = tetra_nodes_tags[i, j]
            python_index = node_index[gmsh_tag]
            elems_numpy[i, j] = python_index

    print(f"Finished transferring: {coordinate.shape[0]} Nodes, {elems_numpy.shape[0]} Tetras")
    return coordinate, elems_numpy

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
    # print(edge_index)
    # print(edge_index.shape)

    # undirected and eliminate repeated edges
    edge_index = to_undirected(edge_index)
    # print(edge_index)
    # print(edge_index.shape)
    edge_index = coalesce(edge_index)

    return edge_index