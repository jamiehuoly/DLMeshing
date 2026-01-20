import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 配置 ---
DATA_FILE = "ground_truth_graph.pt"
MODEL_PATH = os.path.join("trained_models", "gnn_model.pth")
HIDDEN_CHANNELS = 128


# ==========================================
# 1. 必须重新定义模型类 (以便加载权重)
# ==========================================
class MeshRefinementGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, HIDDEN_CHANNELS)
        self.conv2 = SAGEConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.conv3 = SAGEConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        self.conv4 = SAGEConv(HIDDEN_CHANNELS, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)  # No activation
        return x


def verify_results():
    print(f"🕵️‍♂️ 正在加载模型和数据进行最终验收...")

    # 1. 加载数据
    if not os.path.exists(DATA_FILE):
        print("❌ 数据文件不存在")
        return
    data = torch.load(DATA_FILE, weights_only=False)

    # 2. 加载模型
    if not os.path.exists(MODEL_PATH):
        print("❌ 模型文件不存在")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MeshRefinementGNN(in_channels=data.x.shape[1], out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()  # 切换到评估模式 (关掉 Dropout)

    print("✅ 模型加载成功！正在推理...")

    # 3. 推理 (Inference)
    data = data.to(device)
    with torch.no_grad():
        # 获取预测值 (Pred)
        pred_log = model(data.x, data.edge_index)
        # 获取真实值 (Ground Truth)
        true_log = data.y[:, 4].view(-1, 1)

    # 转回 CPU 方便绘图
    pred_np = pred_log.cpu().numpy().flatten()
    true_np = true_log.cpu().numpy().flatten()

    # ------------------------------------------------
    # 4. 第一关：统计验证 (散点图)
    # ------------------------------------------------
    print("\n📊 生成统计散点图...")
    plt.figure(figsize=(6, 6))
    plt.scatter(true_np, pred_np, alpha=0.3, s=2, c='blue')
    # 画 y=x 红线
    min_val = min(true_np.min(), pred_np.min())
    max_val = max(true_np.max(), pred_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    plt.xlabel("Ground Truth (Log Size)")
    plt.ylabel("Model Prediction (Log Size)")
    plt.title(f"Correlation Check\nMSE: {np.mean((true_np - pred_np) ** 2):.5f}")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ------------------------------------------------
    # 5. 第二关：物理视觉验证 (3D 云图对比)
    # ------------------------------------------------
    print("\n👁️ 生成 3D 对比图 (Truth vs Pred vs Error)...")

    # 计算绝对误差
    error = np.abs(true_np - pred_np)

    # 准备 PyVista 网格
    cloud = pv.PolyData(data.pos.cpu().numpy())
    cloud.point_data["Ground_Truth"] = true_np
    cloud.point_data["Prediction"] = pred_np
    cloud.point_data["Abs_Error"] = error
    cloud_middle = cloud.copy()
    cloud_right = cloud.copy()

    # 创建 1行3列 的对比窗口
    plotter = pv.Plotter(shape=(1, 3), window_size=[1800, 600])

    # [左图] 真实值 (老师的答案)
    plotter.subplot(0, 0)
    plotter.add_text("1. Ground Truth (Target)\nBlue=Dense, Red=Coarse", font_size=10)
    plotter.add_mesh(cloud, scalars="Ground_Truth", cmap="jet", point_size=5)

    # [中图] 预测值 (学生的答案)
    plotter.subplot(0, 1)
    plotter.add_text("2. Model Prediction", font_size=10)
    plotter.add_mesh(cloud_middle, scalars="Prediction", cmap="jet", point_size=5)

    # [右图] 误差图 (哪里学得不好)
    plotter.subplot(0, 2)
    plotter.add_text("3. Absolute Error (Diff)", font_size=10)
    plotter.add_mesh(cloud_right, scalars="Abs_Error", cmap="coolwarm", point_size=5, clim=[0, 0.5])
    # clim 设置为 0 到 0.5，让误差明显的地方变红

    plotter.link_views()
    print("✅ 窗口已打开！请依照下方的'验收标准'进行检查。")
    plotter.show()


if __name__ == "__main__":
    verify_results()