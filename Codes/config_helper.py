import json
import os
import sys
import gmsh

CONFIG_FILE = "case_config.json"

# This detection is for vascular problems. It divides plane elements and curved elements.
def visualise_surfaces(geometry_file):
    if not os.path.exists(geometry_file):
        print(f"Error! Could not find geometry file: {geometry_file}")
        return

    gmsh.initialize()
    gmsh.open(geometry_file)
    surfaces = gmsh.model.getEntities(2)
    print("\n" + "-" * 60)
    print("Now providing auto detected predictions:")
    for dim, tag in surfaces:
        stype = gmsh.model.getType(dim, tag)
        role = "Wall?"
        if stype == "Plane":
            role = "Inlet/Outlet?"

        if stype == "Plane":
            print(f"\033[1;32m{tag:<12} | {stype:<10} | {role}\033[0m")  # Green font
        else:
            print(f"{tag:<12} | {stype:<10} | {role}")
            pass

    gmsh.option.setNumber("Geometry.Surfaces", 1)  # 必须显示面
    gmsh.option.setNumber("Geometry.SurfaceType", 2)  # 2 = Filled (实体填充)
    gmsh.option.setNumber("Geometry.Lines", 1)  # 显示边框线
    gmsh.option.setNumber("Geometry.SurfaceLabels", 1)  # 开启 ID 显示
    gmsh.option.setNumber("General.BackgroundGradient", 0)  # 关闭渐变背景
    gmsh.option.setColor("General.Background", 50, 50, 50)  # 深灰色背景
    gmsh.option.setColor("Geometry.Color.Lines", 0, 0, 0)# 线条设为 黑色 (增加轮廓感)

    # 屏蔽日志
    gmsh.option.setNumber("General.Verbosity", 0)

    print("\n" + "Window opening to show the geometry... Feel free to rotate the model.")
    print("Clicking 'Tools' -> 'Visibility' to view more surface ids in detail.")
    gmsh.fltk.run()
    gmsh.finalize()

if __name__ == "__main__":
    # python config_helper.py model.step
    if CONFIG_FILE and os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
    geo_file = config.get("geometry_file", "")

    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        visualise_surfaces(input_file)
    elif geo_file:

        print(f"\nDetected defined geometry file: {geo_file} in case_config.json, start reading...\n")
        visualise_surfaces(geo_file)
    else:
        default_file = "elbow.step"
        visualise_surfaces(default_file)