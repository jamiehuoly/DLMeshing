import torch
import numpy as np
import pyvista as pv
import os

# --- 配置 ---
DATA_FILE = "ground_truth_graph.pt"


def check_visual_side_by_side_fixed():
    print(f"🕵️‍♂️ 正在启动修复后的对比窗口...")

    if not os.path.exists(DATA_FILE):
        print("❌ 文件不存在！")
        return

    data = torch.load(DATA_FILE, weights_only=False)

    # 准备数据
    grad_feat = data.x[:, 7].numpy()
    target_size = data.y[:, 4].numpy()

    # === 关键步骤：创建两个独立的网格对象 ===
    # 1. 左图专用的网格
    cloud_left = pv.PolyData(data.pos.numpy())
    cloud_left.point_data["Input_Gradient"] = grad_feat

    # 2. 右图专用的网格 (深拷贝)
    # 这样我们在右边操作时，绝对不会影响左边
    cloud_right = cloud_left.copy()
    cloud_right.point_data["Target_Size"] = target_size

    # 创建画板 (1行2列)
    plotter = pv.Plotter(shape=(1, 2), window_size=[1600, 800])

    # --- 左窗口：Input (Gradient) ---
    plotter.subplot(0, 0)
    plotter.add_text("Gradient Norm", font_size=10)
    # 使用 cloud_left
    plotter.add_mesh(cloud_left, scalars="Input_Gradient", cmap="jet",
                     render_points_as_spheres=True, point_size=6)  # 这里的截断现在一定生效

    # --- 右窗口：Target (Size) ---
    plotter.subplot(0, 1)
    plotter.add_text("Target Size Ratio (log10)", font_size=10)
    # 使用 cloud_right
    plotter.add_mesh(cloud_right, scalars="Target_Size", cmap="jet",
                     render_points_as_spheres=True, point_size=6)

    # 开启视角同步 (这功能太重要了，一定要保留)
    plotter.link_views()

    print("✅ 修复完成！现在左图应该是红色的，右图是蓝色的。")
    plotter.show()


if __name__ == "__main__":
    check_visual_side_by_side_fixed()