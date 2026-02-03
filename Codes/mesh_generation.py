import os
import gmsh
import utils

def generate_mesh_from_geometry(config):
    step_file = config.get("geometry_file")
    output_coarse_mesh_filename = config.get("output_coarse_mesh_file", "default_coarse_mesh.msh")
    output_path = os.path.join(config["work_dir"], output_coarse_mesh_filename)
    if not step_file:
        print("Error while reading geometry file!")
        return False

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 3)
    gmsh.open(step_file)

    # 清除旧的物理组
    gmsh.model.removePhysicalGroups()
    gmsh.model.occ.synchronize()
    utils.gmsh_option_setting()

    inlet_tags, outlet_tags, wall_tags = utils.get_inlet_outlet_wall_tags(config)

    volumes = gmsh.model.getEntities(3)
    res = utils.add_physical_group(inlet_tags, outlet_tags, wall_tags, volumes)
    if not res:
        print("error when adding physical group, system exit!")
        return False

    gmsh.model.mesh.generate(3)
    gmsh.write(output_path)
    gmsh.finalize()
    return True
