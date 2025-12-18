import gmsh
import gmsh_to_python

# model import and initial 3D mesh generation
gmsh.initialize()
gmsh.model.add("elbow")
gmsh.open("elbow.step")
gmsh.model.mesh.generate(3)

# results from transform
coordinate, elems_numpy = gmsh_to_python.gmsh_tag_transform()
print(coordinate, elems_numpy)