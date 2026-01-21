import gmsh
import os
import sys
import utils

GEOMETRY_FILE = "elbow.step"
POS_FILE = "target_size_field.pos"
OUTPUT_MESH_FILE = "GNN_optimized_mesh.msh"

def generate_mesh_from_pos():
    if not os.path.exists(GEOMETRY_FILE):
        print(f"❌ 错误: 找不到几何文件 {GEOMETRY_FILE}")
        return
    if not os.path.exists(POS_FILE):
        print(f"❌ 错误: 找不到尺寸场文件 {POS_FILE}")
        return

    gmsh.initialize()
    gmsh.open(GEOMETRY_FILE)
    gmsh.merge(POS_FILE)
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

    gmsh.write(OUTPUT_MESH_FILE)
    print(f"3D mesh is generated and saved to {OUTPUT_MESH_FILE}")


    element_types = gmsh.model.mesh.getElementTypes()
    for t in element_types:
        name = gmsh.model.mesh.getElementProperties(t)[0]
        if "Tet" in name: # Tetrahedron
            num = gmsh.model.mesh.getElementsByType(t)[0].size
            print(f"Number of {name}: {num}")

    # pop-ups
    # gmsh.fltk.run()
    gmsh.finalize()


if __name__ == "__main__":
    generate_mesh_from_pos()