import glob
import sys
import gmsh
import numpy as np
import pyvista as pv
import os
import torch
from scipy.interpolate import griddata
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, coalesce

def create_tg_data(x_features, edge_index, y, pos):
    if not torch.is_tensor(x_features):
        x_features = torch.tensor(x_features, dtype=torch.float)
    graph_data = Data(x=x_features, edge_index=edge_index, y=y, pos=pos)
    return graph_data

def add_physical_group(inlet_tags, outlet_tags, wall_tags, volumes):
    if inlet_tags:
        gmsh.model.addPhysicalGroup(2, inlet_tags, tag=1, name="inlet")
    if outlet_tags:
        gmsh.model.addPhysicalGroup(2, outlet_tags, tag=2, name="outlet")
    if wall_tags:
        gmsh.model.addPhysicalGroup(2, wall_tags, tag=3, name="wall")

    if not volumes:
        print("No volumes found in model, please check!")
        gmsh.finalize()
        return False
    volume_tags = [v[1] for v in volumes]
    gmsh.model.addPhysicalGroup(3, volume_tags, tag=100, name="fluid")
    return True

def gmsh_option_setting():
    gmsh.option.setNumber("Mesh.ElementOrder", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.option.setNumber("Mesh.Binary", 0)  # 0 表示 ASCII

def gmsh_open(filename):
    try:
        gmsh.open(filename)
    except:
        print(f"Cannot find file: {filename}")
        gmsh.finalize()
        sys.exit()

def define_mesh_boundaries(surfaces):
    inlet_tmp = []
    outlet_tmp = []
    wall_tmp = []
    TOL = 1e-4
    for s in surfaces:
        tag = s[1]
        geometry_type = gmsh.model.getType(2, tag)
        centroid = gmsh.model.occ.getCenterOfMass(2, tag)
        # Here we are assuming all cases are vascular cases, there should only be 2 planes
        if geometry_type == "Plane":
            x, y, z = centroid[0], centroid[1], centroid[2]
            if abs(x) < TOL or abs(y) < TOL or abs(z) < TOL:
                print(f"-> 发现 Inlet (ID {tag}): 位于 {centroid} (坐标轴面上)")
                inlet_tmp.append(tag)
            else:
                print(f"-> 发现 Outlet (ID {tag}): 位于 {centroid}")
                outlet_tmp.append(tag)
        else:
            wall_tmp.append(tag)
    return inlet_tmp, outlet_tmp, wall_tmp

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

def get_refine_coarse_points(all_geometric_points, yaxis_refine_cutoff, zaxis_refine_cutoff):
    points_to_refine = []
    points_to_coarsen = []
    for dim, tag in all_geometric_points:
        # obtain coordinates of all points
        coord = gmsh.model.getValue(dim, tag, [])
        y_coord = coord[1]
        z_coord = coord[2]

        # select the nodes in the refinement area
        if z_coord > zaxis_refine_cutoff and y_coord < yaxis_refine_cutoff:
            points_to_refine.append((dim, tag))
        else:
            points_to_coarsen.append((dim, tag))
    return points_to_refine, points_to_coarsen

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

def get_latest_vtk(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime)
    return files[-1]

def process_vtk_to_graph(vtk_path):
    try:
        mesh = pv.read(vtk_path)
    except Exception as e:
        print(f"Failed to load vtk file: {vtk_path}; Exception is: {e}")
        return None
    print(f"   - 节点数量: {mesh.n_points}")
    print(f"   - 单元数量: {mesh.n_cells}")

    # Normalisation
    raw_pos = mesh.points  # numpy array
    centroid = np.mean(raw_pos, axis=0)
    max_dist = np.max(np.linalg.norm(raw_pos - centroid, axis=1))
    pos_normalized = (raw_pos - centroid) / max_dist
    print(f"position has been normalized, scale factor is: {max_dist:.4f}")
    x_pos = torch.tensor(pos_normalized, dtype=torch.float)
    print(mesh.point_data)

    if 'U' not in mesh.point_data or "p" not in mesh.point_data:
        print(f"No velocity field U or pressure field p in VTK file, please check! "
              f"Keys currently exist: {mesh.point_data.keys()}")
        return None

    # Labeling
    # pyvista is able to compute derivatives of unstructured mesh
    gradients = mesh.compute_derivative(scalars="U", gradient="grad_U")
    grad_data = gradients.point_data['grad_U']  # Shape (N, 9)
    grad_tensor = grad_data.reshape(-1, 3, 3)

    # Frobenius Norm
    # Larger means flow is more complex --> need to be refined
    error_indicator = np.linalg.norm(grad_tensor, axis=(1, 2))

    # Label Y
    # y[:, 0] = P
    # y[:, 1:4] = U
    # y[:, 4] = Error Indicator
    p_data = mesh.point_data['p']
    u_data = mesh.point_data['U']
    y = torch.tensor(np.column_stack((p_data, u_data, error_indicator)), dtype=torch.float)

    # Normalisation
    feat_p = torch.tensor(p_data, dtype=torch.float).view(-1, 1)
    feat_u = torch.tensor(u_data, dtype=torch.float)
    # Adding 1e-8 防止除0报错
    feat_p = (feat_p - feat_p.mean()) / (feat_p.std() + 1e-8)
    feat_u = (feat_u - feat_u.mean(dim=0)) / (feat_u.std(dim=0) + 1e-8)

    # Add 10% noise to fine CFD results, pretend to be coarse CFD
    noise_level = 0.1
    feat_p_noisy = feat_p + torch.randn_like(feat_p) * noise_level
    feat_u_noisy = feat_u + torch.randn_like(feat_u) * noise_level
    x_features = torch.cat([x_pos, feat_p_noisy, feat_u_noisy], dim=1)

    print(f"   - 标签构建完成. Y Shape: {y.shape}")
    print(f"   - ⚡ 最大梯度模长 (Error Indicator): {error_indicator.max():.4f}")
    if error_indicator.max() < 1e-3:
        print("Warning: gradient is small, flow might be slow or outliers exist.")

    # Graph Topography
    print("   - 正在构建图连接 (这可能需要几秒钟)...")
    edges = mesh.extract_all_edges()
    # edges.lines 的存储格式非常奇葩，是 VTK 的标准：
    # [2, 点A, 点B, 2, 点C, 点D, ...]
    # 这里的 '2' 代表这条线由2个点组成。我们需要把这个 '2' 扔掉。
    # .reshape(-1, 3): 把它变成 N行3列 -> [[2, A, B], [2, C, D], ...]
    # [:, 1:]: 取所有行，但扔掉第0列(那个2) -> [[A, B], [C, D], ...
    lines = edges.lines.reshape(-1, 3)[:, 1:]
    src = lines[:, 0]
    dst = lines[:, 1]

    # 转为 PyG 需要的 (2, E) 格式，且是双向边
    edge_index = torch.tensor(np.vstack((
        np.concatenate([src, dst]),
        np.concatenate([dst, src])
    )), dtype=torch.long)
    print(f"   - 图构建完成. 边数量: {edge_index.shape[1]}")
    return x_features, edge_index, y, x_pos


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