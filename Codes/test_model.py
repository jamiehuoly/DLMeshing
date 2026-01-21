import torch
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
import os

from GraphSAGE import MeshRefinementGNN

DATA_FILE = "ground_truth_graph.pt"
MODEL_PATH = os.path.join("trained_models", "gnn_model.pth")
HIDDEN_CHANNELS = 128

def verify_results():
    if not os.path.exists(DATA_FILE):
        print(f"Cannot find data file: {DATA_FILE}")
        return
    data = torch.load(DATA_FILE, weights_only=False)

    if not os.path.exists(MODEL_PATH):
        print(f"Cannot find model: {MODEL_PATH}")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MeshRefinementGNN(in_channels=data.x.shape[1], out_channels=1).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("Inferencing....")
    data = data.to(device)
    with torch.no_grad():
        pred_log = model(data.x, data.edge_index)
        true_log = data.y[:, 4].view(-1, 1)


    pred_np = pred_log.cpu().numpy().flatten()
    true_np = true_log.cpu().numpy().flatten()

    print("Plotting Graphs....")
    plt.figure(figsize=(6, 6))
    plt.scatter(true_np, pred_np, alpha=0.3, s=2, c='blue')

    min_val = min(true_np.min(), pred_np.min())
    max_val = max(true_np.max(), pred_np.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

    plt.xlabel("Ground Truth (Log Size)")
    plt.ylabel("Model Prediction (Log Size)")
    plt.title(f"Correlation Check\nMSE: {np.mean((true_np - pred_np) ** 2):.5f}")
    plt.grid(True)
    plt.legend()
    plt.show()

    error = np.abs(true_np - pred_np)

    cloud = pv.PolyData(data.pos.cpu().numpy())
    cloud.point_data["Ground_Truth"] = true_np
    cloud.point_data["Prediction"] = pred_np
    cloud.point_data["Abs_Error"] = error
    cloud_middle = cloud.copy()
    cloud_right = cloud.copy()

    plotter = pv.Plotter(shape=(1, 3), window_size=[1800, 600])
    plotter.subplot(0, 0)
    plotter.add_text("1. Ground Truth (Target)\nBlue=Dense, Red=Coarse", font_size=10)
    plotter.add_mesh(cloud, scalars="Ground_Truth", cmap="jet", point_size=5)
    plotter.subplot(0, 1)
    plotter.add_text("2. Model Prediction", font_size=10)
    plotter.add_mesh(cloud_middle, scalars="Prediction", cmap="jet", point_size=5)
    plotter.subplot(0, 2)
    plotter.add_text("3. Absolute Error (Diff)", font_size=10)
    plotter.add_mesh(cloud_right, scalars="Abs_Error", cmap="coolwarm", point_size=5, clim=[0, 0.5])

    plotter.link_views()
    plotter.show()


if __name__ == "__main__":
    verify_results()