import gmsh
import sys
import os.path

import torch
from torch_geometric.utils import is_undirected

import utils

# 1. model import and initial 3D mesh generation
gmsh.initialize()
gmsh.model.add("elbow")
utils.gmsh_open("elbow.step")

# gmsh.fltk.run()
gmsh.model.occ.synchronize()
surfaces = gmsh.model.getEntities(2)

# Define Boundaries
inlet_tags, outlet_tags, wall_tags = utils.define_mesh_boundaries(surfaces)

# Define Physical Groups
## Inlet & Outlet
if inlet_tags:
    gmsh.model.addPhysicalGroup(2, inlet_tags, tag=1, name="inlet")
if outlet_tags:
    gmsh.model.addPhysicalGroup(2, outlet_tags, tag=2, name="outlet")
if wall_tags:
    gmsh.model.addPhysicalGroup(2, wall_tags, tag=3, name="wall")

## Entire 3D Flow Field
volumes = gmsh.model.getEntities(3)
if not volumes:
    print("No volumes found in model, please check!")
    gmsh.finalize()
    sys.exit()
volume_tags = [v[1] for v in volumes]
gmsh.model.addPhysicalGroup(3, volume_tags, tag=100, name="fluid")

gmsh.option.setNumber("Mesh.ElementOrder", 1)
gmsh.option.setNumber("Mesh.MeshSizeFactor", 0.5)
gmsh.option.setNumber("Mesh.Optimize", 1)
gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)

print("Generating 3D mesh...")
gmsh.model.mesh.generate(3)

gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.option.setNumber("Mesh.Binary", 0) # 0 表示 ASCII
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
