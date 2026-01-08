import gmsh
import sys
import os.path

import torch
from torch_geometric.utils import is_undirected

import utils

GENERATE_FINE_MESH = True

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

# if GENERATE_FINE_MESH:
#     # get model edges
#     bbox = gmsh.model.getBoundingBox(-1, -1)
#
#     xmin, ymin, zmin = bbox[0], bbox[1], bbox[2]
#     xmax, ymax, zmax = bbox[3], bbox[4], bbox[5]

print("Generating 3D mesh...")
gmsh.model.mesh.generate(3)
gmsh.write("elbow.msh")

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
