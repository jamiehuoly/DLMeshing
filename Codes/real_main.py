import os
import sys
import torch
import utils
from mesh_generation import generate_mesh_from_geometry
from inference_to_position import generate_size_field
from position_to_gmsh import generate_mesh_from_pos


GENERATE_FINE_MESH = True
CONFIG_FILE = "case_config.json"
WORKING_DIRECTORY = "case_working"
MESH_FILE_NAME = "elbow.msh"
VTK_FILE_PATTERN = "VTK_fine/DLMeshing_fine_*.vtk"
GRAPH_DATA_FILE_NAME = "ground_truth_graph.pt"
MODEL_SAVE_PATH = "trained_models"
MODEL_SAVE_NAME = "gnn_model.pth"
POS_FILE_NAME = "target_size_field.pos"
LEARNING_RATE = 0.001
EPOCHS = 2000

if len(sys.argv) > 1:
    MODE = sys.argv[1]
    MODE = MODE.lower()
else:
    MODE = "inference"

# read config file & generate coarse mesh
generate_coarse = generate_mesh_from_geometry(CONFIG_FILE)
if not generate_coarse:
    print(f"Failed to generate initial mesh!")
    utils.safe_exit()

## ToDo: OpenFOAM

vtk_file = utils.get_latest_vtk(VTK_FILE_PATTERN)
print(f"VTK File Name: {vtk_file}")

## ToDo: Training
# x_features, edge_index, y, pos, L_Char, raw_pos = utils.process_vtk_to_graph(vtk_file, MODE)
# graph_data = utils.create_tg_data(x_features, edge_index, y, pos)
# graph_data.L_Char = torch.tensor(L_Char, dtype=torch.float)
# target_path = os.path.join(WORKING_DIRECTORY, GRAPH_DATA_FILE_NAME)
# torch.save(graph_data, f"{target_path}")
# print(f"Saved graph data to path: {target_path}")

# Inference
config = utils.get_config_json(CONFIG_FILE)
pos_output_path = os.path.join(config["work_dir"], POS_FILE_NAME)
generate_size_field(vtk_file, pos_output_path, MODE)
generate_mesh_from_pos(config, pos_output_path)
print("Finished generating optimized mesh!")




