import os
import sys
import torch
import utils
from foam_operator import OpenFoamAutomator
from GraphSAGE import MeshRefinementGNN
from mesh_generation import generate_mesh_from_geometry
from inference_to_position import generate_size_field
from position_to_gmsh import generate_mesh_from_pos


GENERATE_FINE_MESH = True
CONFIG_FILE = "case_config.json"
MESH_FILE_NAME = "elbow.msh"
INFER_VTK_FILE_PATTERN = "VTK_Coarse/DLMeshing_coarse_*.vtk"
TRAIN_VTK_FILE_PATTERN = "VTK_fine/DLMeshing_fine_*.vtk"
REAL_VTK_FILE_PATTERN = "VTK/*.vtk"
GRAPH_DATA_FILE_NAME = "ground_truth_graph.pt"
MODEL_SAVE_PATH = "models"
MODEL_SAVE_NAME = "gnn_model.pth"
LEARNING_RATE = 0.001
EPOCHS = 2000

# python3 real_main.py train/inference
if len(sys.argv) > 1:
    MODE = sys.argv[1]
    MODE = MODE.lower()
    if MODE != "train" and MODE != "inference":
        print("Invalid mode! Mode can only be 'train' or 'inference'")
        utils.safe_exit()
else:
    # default setting
    MODE = "inference"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config = utils.get_config_json(CONFIG_FILE)
utils.parse_working_dir(config)

# read config file & generate coarse mesh
generate_coarse = generate_mesh_from_geometry(config)
if not generate_coarse:
    print(f"Failed to generate initial mesh!")
    utils.safe_exit()

runner = OpenFoamAutomator(config["work_dir"])
runner.clean_logs()
runner.prepare_compulsory_folders()

runner.import_gmsh(config["output_coarse_mesh_file"])
runner.scale_mesh(0.001)  # mm to m
runner.check_mesh()

runner.update_boundary_conditions(config)

runner.run_solver("foamRun")
runner.export_vtk()
runner.create_dummy_foam()
runner.clean_redundants()

if MODE == "train":
    vtk_file = utils.get_latest_vtk(TRAIN_VTK_FILE_PATTERN)
    x_features, edge_index, y, pos, L_Char, raw_pos = utils.process_vtk_to_graph(vtk_file, MODE)
    graph_data = utils.create_tg_data(x_features, edge_index, y, pos)
    graph_data.L_Char = torch.tensor(L_Char, dtype=torch.float)
    target_path = os.path.join(config["work_dir"], GRAPH_DATA_FILE_NAME)
    torch.save(graph_data, f"{target_path}")
    print(f"Saved graph data to path: {target_path}")

    data, train_idx, test_idx = utils.train_data_process(target_path, device)

    # 主要训练 LC
    target = data.y[:, 4].view(-1, 1)

    model = MeshRefinementGNN(in_channels=data.x.shape[1], out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.SmoothL1Loss(beta=1.0)  # Huber loss (分段函数), combines L1 loss (MSE) & L2 loss (MAE)
    loss_history = []

    print(f"Start training {EPOCHS} Epochs...")
    model.train()

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output = model(data.x, data.edge_index)

        loss = criterion(output[train_idx], target[train_idx])
        loss.backward()
        optimizer.step()
        # Carry out test loss every 100 epoch
        if (epoch + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                test_loss = criterion(output[test_idx], target[test_idx])
            model.train()

            loss_history.append(loss.item())
            print(
                f"Epoch {epoch + 1:04d} | Train Loss (SmoothL1): {loss.item():.6f} | Test Loss: {test_loss.item():.6f}")

    # The trained model will be saved to the output directory instead of going directly to 'model' directory
    # The 'models' directory is for formally trained model
    model_save_directory = os.path.join(config["work_dir"], MODEL_SAVE_PATH)
    model_save_path = os.path.join(model_save_directory, MODEL_SAVE_NAME)
    if not os.path.exists(model_save_directory):
        os.makedirs(model_save_directory)
    torch.save(model.state_dict(), model_save_path)
    print(f"\nModel saved to: ./{model_save_path}")

else:
    # Inference
    # The model used for inference will be the legit one from 'models' directory
    vtk_path_pattern = os.path.join(config["work_dir"], REAL_VTK_FILE_PATTERN)
    vtk_file = utils.get_latest_vtk(vtk_path_pattern)
    generate_size_field(vtk_file, config, MODE)
    generate_mesh_from_pos(config)
    print("Finished generating optimized mesh!")




