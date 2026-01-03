import gmsh
import numpy as np
import torch
from scipy.interpolate import griddata
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce

def create_tg_data(x_features, edge_index):
    coordinate_tensor = torch.tensor(x_features, dtype=torch.float)
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

def get_tetrahedral_edges(coordinate, elems_numpy):
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

def enrich_features(fine_coordinates, coarse_data_path):
    """
    fine_coordinates: coordinate output of gmsh_tag_transform, with a shape of [N, 3]
    coarse_data_path: coarse CFD result file path (txt, csv, vtk)
    """
    # 1. load CFD results
    # N_coarse x 7: [x, y, z, u, v, w, p]

    # 假设有 5000 个粗糙点, N_coarse = coarse_nodes.shape[0]
    N_coarse = 5000
    # To-do np.load(coarse_data_path)
    coarse_pos = np.random.rand(N_coarse, 3)  # 粗坐标
    coarse_vel = np.random.rand(N_coarse, 3)  # 粗速度
    coarse_p = np.random.rand(N_coarse, 1)  # 粗压力

    # 2. To-do: Calculate coarse derivatives

    # 3. insert values to coordinate list
        # velocity
    interp_vel = griddata(coarse_pos, coarse_vel, fine_coordinates, method='linear')
        # 处理插值产生的 NaN (边界外)
    interp_vel = np.nan_to_num(interp_vel, nan=0.0)

        # pressure
    interp_p = griddata(coarse_pos, coarse_p, fine_coordinates, method='linear')
    interp_p = np.nan_to_num(interp_p, nan=0.0)

    # 4. normalisation
        # coordinate
    norm_p, norm_pos, norm_vel = normalisation(fine_coordinates, interp_p, interp_vel)

    # 5. Stacking 拼接
    # final = [Norm_X, Norm_Y, Norm_Z, Norm_U, Norm_V, Norm_W, Norm_P]
    # dimensions: [N, 3] + [N, 3] + [N, 1] = [N, 7]
    combined_features = np.hstack([norm_pos, norm_vel, norm_p])

    return combined_features


def normalisation(fine_coordinates, interp_p, interp_vel):
    centroid = np.mean(fine_coordinates, axis=0)
    scale_pos = np.max(np.abs(fine_coordinates - centroid))
    norm_pos = (fine_coordinates - centroid) / scale_pos

    # velocity
    max_vel = np.max(np.linalg.norm(interp_vel, axis=1))
    norm_vel = interp_vel / (max_vel + 1e-8)

    # pressure
    max_p = np.max(np.abs(interp_p))
    norm_p = interp_p / (max_p + 1e-8)

    return norm_p, norm_pos, norm_vel