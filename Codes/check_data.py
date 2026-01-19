import torch
import numpy as np
import pyvista as pv

# --- 配置 ---
DATA_FILE = "ground_truth_graph.pt"  # 你刚刚生成的文件名


def check_dataset():
    print(f"🕵️‍♂️ 正在检查数据文件: {DATA_FILE} ...\n")

    if not os.path.exists(DATA_FILE):
        print("❌ 文件不存在！请先运行处理脚本。")
        return

    # 1. 加载数据
    data = torch.load(DATA_FILE, weights_only=False)
    print("=" * 40)
    print("📊 维度概览 (Shape Check)")
    print("=" * 40)

    num_nodes = data.x.shape[0]
    print(f"1. 节点数量 (N): {num_nodes}")
    print(f"2. 输入特征 X 形状: {data.x.shape} (预期: [N, 7])")
    print(f"3. 标签目标 Y 形状: {data.y.shape} (预期: [N, 5])")
    print(f"4. 边连接 Edge_index: {data.edge_index.shape} (预期: [2, E])")

    # 检查边是否合理
    avg_degree = data.edge_index.shape[1] / num_nodes
    print(f"5. 平均度数 (Degree): {avg_degree:.2f}")
    if avg_degree < 2:
        print("   ⚠️ 警告: 平均连接数过低，图可能太稀疏了！")

    print("\n" + "=" * 40)
    print("🧮 统计数值检查 (Statistical Check)")
    print("=" * 40)

    # --- 检查 NaN/Inf (致命错误) ---
    has_nan_x = torch.isnan(data.x).any()
    has_inf_x = torch.isinf(data.x).any()
    has_nan_y = torch.isnan(data.y).any()

    if has_nan_x or has_inf_x or has_nan_y:
        print("❌❌❌ 严重错误: 数据中包含 NaN 或 Inf！")
        print(f"   X has NaN: {has_nan_x}, Inf: {has_inf_x}")
        print(f"   Y has NaN: {has_nan_y}")
        return  # 直接退出，不用看了
    else:
        print("✅ 数据清洁度: 通过 (无 NaN/Inf)")

    # --- 检查输入 X 的归一化情况 ---
    # X columns: [0,1,2]=Pos, [3]=P_noisy, [4,5,6]=U_noisy
    x_pos = data.x[:, 0:3]
    x_p = data.x[:, 3]
    x_u = data.x[:, 4:7]

    print(f"\n[Input X] 几何坐标 (0-2列):")
    print(f"   Min: {x_pos.min():.4f}, Max: {x_pos.max():.4f}")
    if x_pos.max() > 1.5 or x_pos.min() < -1.5:
        print("   ⚠️ 警告: 坐标归一化可能未完成，数值依然很大。")
    else:
        print("   ✅ 范围正常 ([-1, 1] 附近)")

    print(f"[Input X] 压力 P (第3列):")
    print(f"   Mean: {x_p.mean():.4f}, Std: {x_p.std():.4f}")
    if abs(x_p.mean()) > 0.1 or abs(x_p.std() - 1.0) > 0.2:
        print("   ⚠️ 警告: P 未标准化 (Mean!=0 或 Std!=1)。GNN 可能难收敛。")
    else:
        print("   ✅ 标准化正常 (Mean~0, Std~1)")

    # --- 检查标签 Y 的有效性 ---
    # Y columns: [0]=P, [1-3]=U, [4]=Error
    y_error = data.y[:, 4]
    print(f"\n[Label Y] 误差指示器 (最后一列):")
    print(f"   Max Gradient: {y_error.max():.4f}")
    print(f"   Mean Gradient: {y_error.mean():.4f}")

    if y_error.max() < 1e-4:
        print("   ❌ 错误: 最大梯度几乎为0。Label 计算失败，或者流场是静止的。")
    elif y_error.max() > 10000:
        print("   ⚠️ 警告: 梯度极大。虽然物理上可能，但容易导致梯度爆炸。")
    else:
        print("   ✅ 梯度范围合理。")

    print("\n" + "=" * 40)
    print("👁️ 视觉抽样检查 (Visual Check)")
    print("=" * 40)
    print("正在启动 PyVista 可视化窗口...")
    print("请检查：高亮的红色区域是否位于弯头内侧？")

    # 还原为 PyVista 对象进行查看
    # 我们只看 Error Indicator，看看它长得对不对
    cloud = pv.PolyData(data.pos.numpy())
    cloud.point_data["Error_Label"] = y_error.numpy()
    cloud.point_data["Input_P_Noisy"] = x_p.numpy()

    # 绘制
    plotter = pv.Plotter(shape=(1, 2))  # 左右两个子图

    plotter.subplot(0, 0)
    plotter.add_text("Input X: Noisy Pressure", font_size=10)
    plotter.add_mesh(cloud, scalars="Input_P_Noisy", cmap="coolwarm", render_points_as_spheres=True, point_size=5)

    plotter.subplot(0, 1)
    plotter.add_text("Target Y: Error Indicator (Gradient)", font_size=10)
    plotter.add_mesh(cloud, scalars="Error_Label", cmap="jet", render_points_as_spheres=True, point_size=5)

    plotter.link_views()  # 联动旋转
    plotter.show()


import os

if __name__ == "__main__":
    check_dataset()