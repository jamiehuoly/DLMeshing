import gmsh
import sys
import os.path

import torch
from torch_geometric.utils import is_undirected

import utils

GENERATE_FINE_MESH = True
MESH_FILE_NAME = "elbow.msh"

# 1. model import and initial 3D mesh generation
gmsh.initialize()
gmsh.model.add("elbow")
utils.gmsh_open("elbow.step")

# gmsh.fltk.run()
gmsh.model.occ.synchronize()
surfaces = gmsh.model.getEntities(2)
volumes = gmsh.model.getEntities(3)

# Define Boundaries
inlet_tags, outlet_tags, wall_tags = utils.define_mesh_boundaries(surfaces)

# Define Physical Groups
## Inlet, Outlet, wall, flow field
res = utils.add_physical_group(inlet_tags, outlet_tags, wall_tags, volumes)
if not res:
    print("error when adding physical group, system exit!")
    sys.exit(0)

utils.gmsh_option_setting()

if GENERATE_FINE_MESH:
    # refinement parameters / characteristic length
    LC_FINE = 1.2 # fine section
    LC_COARSE = 4.2  # coarse section

    # let 30% coarse, 70% fine
    bbox = gmsh.model.getBoundingBox(-1, -1)
    ymin, ymax = bbox[1], bbox[4]
    zmin, zmax = bbox[2], bbox[5]
    yaxis_refine_cutoff = ymax + (ymin - ymax) * 0.2 # the y-coordinate is inversed
    zaxis_refine_cutoff = zmin + (zmax - zmin) * 0.2

    # obtain all points and their information
    all_geometric_points = gmsh.model.getEntities(0)

    points_to_refine = []
    points_to_coarsen = []

    print(f"检测到模型共有 {len(all_geometric_points)} 个几何顶点。")

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

    # use gmsh setSize to set the characteristic length of nodes in refinement section
    if points_to_refine:
        gmsh.model.mesh.setSize(points_to_refine, LC_FINE)
        print(f"已强制设置 {len(points_to_refine)} 个底部顶点的尺寸为 {LC_FINE}")

    # use gmsh setSize to set the characteristic length of nodes in coarse section
    if points_to_coarsen:
        gmsh.model.mesh.setSize(points_to_coarsen, LC_COARSE)
        print(f"已强制设置 {len(points_to_coarsen)} 个顶部顶点的尺寸为 {LC_COARSE}")

    MESH_FILE_NAME = "fine_elbow.msh"

print("Generating 3D mesh...")
gmsh.model.mesh.generate(3)
gmsh.write(MESH_FILE_NAME)

# 2. transformed results
### should be sure that the coordinates will be consistent to the CFD results
coordinate, elems_numpy = utils.gmsh_tag_transform()
# print(coordinate, elems_numpy)
# print(coordinate.shape)

# 3. build torch_geometric data for NN
edges_index = utils.get_tetrahedral_edges(coordinate, elems_numpy)
print(is_undirected(edges_index))

# 4. stack u v w p into the list
features_7d = utils.enrich_features(coordinate, "to/be/filled/path")

# 5. create Data object
graph_data = utils.create_tg_data(features_7d, edges_index)
print(edges_index)
print(edges_index.shape)
print(graph_data)

torch.save(graph_data, "elbow_graph_input.pt")
print("Saved graph data to elbow_graph_input.pt")

gmsh.finalize()

# To do: loss function define & model initialising & training
# To do: Hardware detect algorithm (GPU/CPU)
