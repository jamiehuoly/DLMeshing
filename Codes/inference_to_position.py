import torch
import numpy as np
import os

import utils
from GraphSAGE import MeshRefinementGNN

UNIT_SCALE_FACTOR = 1000.0
COARSE_VTK_PATH = "VTK_Coarse/DLMeshing_coarse_300.vtk"
MODEL_PATH = "trained_models/gnn_model.pth"
OUTPUT_POS_FILE = "target_size_field.pos"

def generate_size_field():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_features, edge_index, _, x_pos, L_char, raw_pos = utils.process_vtk_to_graph(COARSE_VTK_PATH, mode="inference")
    x = x_features.to(device)
    edge_index = edge_index.to(device)

    in_channels = x.shape[1]
    model = MeshRefinementGNN(in_channels=in_channels, out_channels=1).to(device)
    if not os.path.exists(MODEL_PATH):
        print(f"Cannot find model: {MODEL_PATH}")
        return
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    print("Inferencing...")
    with torch.no_grad():
        # 输出是 Log10(Size Ratio)
        pred_log_ratio = model(x, edge_index)

    # 还原为真实物理尺寸: h = 10^(pred) * L_char
    pred_ratio = 10 ** pred_log_ratio.cpu().numpy().flatten()
    target_sizes = pred_ratio * L_char

    # 为了防止生成极其微小的死循环网格，设置一个下限 (Min Size)
    min_size_limit = L_char * 0.001
    target_sizes = np.maximum(target_sizes, min_size_limit)

    print(f"Now writing .pos file: {OUTPUT_POS_FILE}")
    with open(OUTPUT_POS_FILE, "w") as f:
        f.write('View "Background Mesh" {\n')
        for i in range(len(raw_pos)):
            px, py, pz = raw_pos[i] * UNIT_SCALE_FACTOR
            val = target_sizes[i] * UNIT_SCALE_FACTOR
            # Gmsh 格式: SP(x,y,z){value};
            f.write(f"SP({px:.6f},{py:.6f},{pz:.6f}){{{val:.6f}}};\n")
        f.write('};\n')

    print("Finished writing .pos file!")


if __name__ == "__main__":
    generate_size_field()