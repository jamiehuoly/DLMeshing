import gmsh
import os
import sys
import utils

POS_FILE_NAME = "target_size_field.pos"

# Physical groups are excluded
def generate_mesh_from_pos(config):
    geometry_file = config["geometry_file"]
    pos_output_path = os.path.join(config["work_dir"], POS_FILE_NAME)
    if not os.path.exists(geometry_file):
        print(f"Error! Cannot find geometry file {geometry_file}")
        return
    if not os.path.exists(pos_output_path):
        print(f"Error! Cannot find .pos file {pos_output_path}")
        return

    gmsh.initialize()
    gmsh.open(geometry_file)
    gmsh.merge(pos_output_path)
    gmsh.model.removePhysicalGroups()
    gmsh.model.occ.synchronize()
    inlet_tags, outlet_tags, wall_tags = utils.get_inlet_outlet_wall_tags(config)

    volumes = gmsh.model.getEntities(3)
    res = utils.add_physical_group(inlet_tags, outlet_tags, wall_tags, volumes)
    if not res:
        print("Error occurs when adding physical group of final mesh, system exit!")

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
