import torch
import numpy as np
import os

import utils
from GraphSAGE import MeshRefinementGNN

UNIT_SCALE_FACTOR = 1000.0
COARSE_VTK_PATH = "VTK_Coarse/DLMeshing_coarse_300.vtk"
MODEL_PATH = "models/gnn_model.pth"
POS_FILE_NAME = "target_size_field.pos"

def generate_size_field(coarse_vtk_path, config, mode="inference", device="cpu"):
    x_features, edge_index, _, x_pos, L_char, raw_pos = utils.process_vtk_to_graph(coarse_vtk_path, mode=mode)
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
        # output format: Log10(Size Ratio)
        pred_log_ratio = model(x, edge_index)

    # back to real value: h = 10^(pred) * L_char
    pred_ratio = 10 ** pred_log_ratio.cpu().numpy().flatten()
    target_sizes = pred_ratio * L_char

    # 为了防止生成极其微小的死循环网格，设置一个下限 (Min Size)
    min_size_limit = L_char * 0.001
    target_sizes = np.maximum(target_sizes, min_size_limit)

    pos_output_path = os.path.join(config["work_dir"], POS_FILE_NAME)
    print(f"Now writing .pos file to: {pos_output_path}")
    with open(pos_output_path, "w") as f:
        f.write('View "Background Mesh" {\n')
        for i in range(len(raw_pos)):
            px, py, pz = raw_pos[i] * UNIT_SCALE_FACTOR
            val = target_sizes[i] * UNIT_SCALE_FACTOR
            # Gmsh style: SP(x,y,z){value};
            f.write(f"SP({px:.6f},{py:.6f},{pz:.6f}){{{val:.6f}}};\n")
        f.write('};\n')

    print("Finished writing .pos file!")
