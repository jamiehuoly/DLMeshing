import gmsh
import os
import sys
import utils

# GEOMETRY_FILE = "elbow.step"
# POS_FILE = "target_size_field.pos"
# OUTPUT_MESH_FILE = "GNN_optimized_mesh.msh"

# Physical groups are excluded
def generate_mesh_from_pos(config, pos_output_path):
    geometry_file = config["geometry_file"]
    if not os.path.exists(geometry_file):
        print(f"Error! Cannot find geometry file {geometry_file}")
        return
    if not os.path.exists(pos_output_path):
        print(f"Error! Cannot find .pos file {pos_output_path}")
        return

    gmsh.initialize()
    gmsh.open(geometry_file)
    gmsh.merge(pos_output_path)
    background = gmsh.model.mesh.field.add("PostView")

    gmsh.model.mesh.field.setNumber(background, "ViewIndex", 0)
    gmsh.model.mesh.field.setAsBackgroundMesh(background)

    utils.gmsh_option_setting()
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)

    print("Generating 3D mesh....")
    try:
        gmsh.model.mesh.generate(3)
    except Exception as e:
        print(f"Error occurs when generating 3D mesh: {e}")
        gmsh.finalize()
        return
    output_mesh_path = os.path.join(config["work_dir"], config["output_optimized_mesh_file"])
    gmsh.write(output_mesh_path)
    print(f"3D mesh is generated and saved to: {output_mesh_path}")


    # element_types = gmsh.model.mesh.getElementTypes()
    # for t in element_types:
    #     name = gmsh.model.mesh.getElementProperties(t)[0]
    #     if "Tet" in name: # Tetrahedron
    #         num = gmsh.model.mesh.getElementsByType(t)[0].size
    #         print(f"Number of {name}: {num}")

    # pop-ups
    # gmsh.fltk.run()
    gmsh.finalize()
