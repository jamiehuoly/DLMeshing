import torch
import numpy as np
import pyvista as pv
import os

DATA_FILE = "ground_truth_graph.pt"

def check_visual_side_by_side_fixed():
    if not os.path.exists(DATA_FILE):
        print(f"Cannot find data file: {DATA_FILE}")
        return

    data = torch.load(DATA_FILE, weights_only=False)

    grad_feat = data.x[:, 7].numpy()
    target_size = data.y[:, 4].numpy()

    cloud_left = pv.PolyData(data.pos.numpy())
    cloud_left.point_data["Input_Gradient"] = grad_feat
    cloud_right = cloud_left.copy()
    cloud_right.point_data["Target_Size"] = target_size

    plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])
    plotter.subplot(0, 0)
    plotter.add_text("Gradient Norm", font_size=10)
    plotter.add_mesh(cloud_left, scalars="Input_Gradient", cmap="jet",
                     render_points_as_spheres=True, point_size=6)  # 这里的截断现在一定生效
    plotter.subplot(0, 1)
    plotter.add_text("Target Size Ratio (log10)", font_size=10)
    # 使用 cloud_right
    plotter.add_mesh(cloud_right, scalars="Target_Size", cmap="jet",
                     render_points_as_spheres=True, point_size=6)

    plotter.link_views()
    plotter.show()


if __name__ == "__main__":
    check_visual_side_by_side_fixed()