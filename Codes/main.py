import gmsh
import sys
import os.path

import torch
from torch_geometric.utils import is_undirected

import utils

GENERATE_FINE_MESH = True
MESH_FILE_NAME = "elbow.msh"
VTK_FILE_PATTERN = "VTK_fine/DLMeshing_fine_*.vtk"
GRAPH_DATA_FILE_NAME = "ground_truth_graph.pt"

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

# Gmsh universal settings
utils.gmsh_option_setting()

if GENERATE_FINE_MESH:
    # refinement parameters / characteristic length
    # gmsh does not care about units, it only cares about magnitude
    LC_FINE = 1.0 # fine section
    LC_COARSE = 4.3 # coarse section
    x_proportion = 0.4
    y_proportion = 0.1

    # Refinement starts 10% away from inlet/outlet
    bbox = gmsh.model.getBoundingBox(-1, -1)
    ymin, ymax = bbox[1], bbox[4]
    zmin, zmax = bbox[2], bbox[5]
    yaxis_refine_cutoff = ymax + (ymin - ymax) * y_proportion # the y-coordinate is inversed
    zaxis_refine_cutoff = zmin + (zmax - zmin) * x_proportion

    # obtain all points and their information
    all_geometric_points = gmsh.model.getEntities(0)
    points_to_refine, points_to_coarsen = utils.get_refine_coarse_points(all_geometric_points,
                                                                         yaxis_refine_cutoff, zaxis_refine_cutoff)

    # use gmsh setSize to set the characteristic length of nodes in refinement section
    if points_to_refine:
        gmsh.model.mesh.setSize(points_to_refine, LC_FINE)
        print(f"Sizes of {len(points_to_refine)} nodes has been set to {LC_FINE}")

    # use gmsh setSize to set the characteristic length of nodes in coarse section
    if points_to_coarsen:
        gmsh.model.mesh.setSize(points_to_coarsen, LC_COARSE)
        print(f"Sizes of {len(points_to_coarsen)} nodes has been set to {LC_COARSE}")

    MESH_FILE_NAME = "fine_elbow.msh"

print("Generating 3D mesh...")
gmsh.model.mesh.generate(3)
gmsh.write(MESH_FILE_NAME)

# ToDo: Needs to integrate the OpenFoam operations here (gmshToFoam, editing files, foamRun, foamToVTK)

# 3. transform results from VTK files (x and y are 7-dimensional: x,y,z,p,u,v,w)
file = utils.get_latest_vtk(VTK_FILE_PATTERN)
x_features, edge_index, y, pos = utils.process_vtk_to_graph(file)

# 4. create Data object
graph_data = utils.create_tg_data(x_features, edge_index, y, pos)
torch.save(graph_data, f"{GRAPH_DATA_FILE_NAME}")
print(f"Saved graph data to file: {GRAPH_DATA_FILE_NAME}")

gmsh.finalize()

# ToDo: loss function define & model initialising & training
# ToDo: Hardware detect algorithm (GPU/CPU)



