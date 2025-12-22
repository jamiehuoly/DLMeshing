import gmsh
import gmsh_to_numpy
import edges
import utils

# 1. model import and initial 3D mesh generation
gmsh.initialize()
gmsh.model.add("elbow")
gmsh.open("elbow.step")
gmsh.model.mesh.generate(3)

# 2. transformed results
coordinate, elems_numpy = gmsh_to_numpy.gmsh_tag_transform()
# print(coordinate, elems_numpy)
# print(elems_numpy.shape)

# 3. build torch_geometric data for NN
edges_index = edges.get_graph_edges(coordinate, elems_numpy)
graph_data = utils.create_tg_data(coordinate, edges_index)
# print(edges_index[0])
print(graph_data)



