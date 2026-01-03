import gmsh
from torch_geometric.utils import is_undirected

import utils

# 1. model import and initial 3D mesh generation
gmsh.initialize()
gmsh.model.add("elbow")
gmsh.open("elbow.step")
gmsh.model.mesh.generate(3)

# 2. transformed results
coordinate, elems_numpy = utils.gmsh_tag_transform()
# print(coordinate, elems_numpy)
# print(coordinate.shape)

# 3. build torch_geometric data for NN
edges_index = utils.get_graph_edges(coordinate, elems_numpy)
print(is_undirected(edges_index))

graph_data = utils.create_tg_data(coordinate, edges_index)
print(edges_index)
print(edges_index.shape)
print(graph_data)


# To do: loss function define & model initialising & training
# To do: Hardware detect algorithm (GPU/CPU)
