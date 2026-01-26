import os
import gmsh
import utils

def generate_mesh_from_geometry(config_file=None, geometry_file=None):
    inlet_tags = outlet_tags = wall_tags = []
    config = utils.read_geo_config(config_file, geometry_file)
    step_file = config.get("geometry_file", geometry_file)
    output_coarse_mesh_file = config.get("output_coarse_mesh_file", "default_coarse_mesh.msh")
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

    # First branch: Mesh information provided by customers
    if config.get("mode").lower() == "manual" and "boundaries" in config:
        print("Mode is manual! Processing mesh config file....")
        bounds = config["boundaries"]

        if "inlet" in bounds and bounds["inlet"]["surface_ids"]:
            inlet_tags = bounds["inlet"]["surface_ids"]

        if "outlet" in bounds and bounds["outlet"]["surface_ids"]:
            outlet_tags = bounds["outlet"]["surface_ids"]

        if "wall" in bounds and bounds["wall"]["surface_ids"]:
            wall_tags = bounds["wall"]["surface_ids"]

    # Auto Detect
    else:
        print("Carrying out auto detection mode...")
        surfaces = gmsh.model.getEntities(2)
        inlet_tags, outlet_tags, wall_tags = utils.define_mesh_boundaries(surfaces)

    volumes = gmsh.model.getEntities(3)
    res = utils.add_physical_group(inlet_tags, outlet_tags, wall_tags, volumes)
    if not res:
        print("error when adding physical group, system exit!")
        return False

    gmsh.model.mesh.generate(3)
    gmsh.write(output_coarse_mesh_file)
    gmsh.finalize()
    return True


if __name__ == "__main__":
    if os.path.exists("case_config.json"):
        generate_mesh_from_geometry("case_config.json")
    else:
        generate_mesh_from_geometry(geometry_file="elbow.step")