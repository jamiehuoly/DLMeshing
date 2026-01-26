import gmsh
import sys
import os.path

import torch
import matplotlib.pyplot as plt
from torch_geometric.utils import is_undirected
from GraphSAGE import MeshRefinementGNN

import utils

# ToDo: These procedures in main.py should be refracted to individual files
# ToDo: Constant variables should all be defined in a config file rather than .py

MODE = "Train"
GENERATE_FINE_MESH = True
MESH_FILE_NAME = "elbow.msh"
VTK_FILE_PATTERN = "VTK_fine/DLMeshing_fine_*.vtk"
GRAPH_DATA_FILE_NAME = "ground_truth_graph.pt"
MODEL_SAVE_PATH = "trained_models"
MODEL_SAVE_NAME = "gnn_model.pth"
LEARNING_RATE = 0.001
EPOCHS = 2000

# 1. model import and initial 3D mesh generation
gmsh.initialize()
gmsh.model.add("elbow")
utils.gmsh_open("elbow.step")

# gmsh.fltk.run()
gmsh.model.occ.synchronize()
surfaces = gmsh.model.getEntities(2)
volumes = gmsh.model.getEntities(3)

# Define Boundaries
inlet_tags, outlet_tags, wall_tags = utils.define_mesh_boundaries(surfaces)

# Define Physical Groups
## Inlet, Outlet, wall, flow field
## ToDo: BIG PROBLEM: It is impossible for customers to provide CFD information of geometry
res = utils.add_physical_group(inlet_tags, outlet_tags, wall_tags, volumes)
if not res:
    print("error when adding physical group, system exit!")
    utils.safe_exit()

# Gmsh universal settings
utils.gmsh_option_setting()

if GENERATE_FINE_MESH:
    # refinement parameters / characteristic length
    # gmsh does not care about units, it only cares about magnitude
    LC_FINE = 1.0 # fine section
    LC_COARSE = 4.3 # coarse section
    x_proportion = 0.4
    y_proportion = 0.1

    # Refinement starts 10% away from inlet/outlet
    bbox = gmsh.model.getBoundingBox(-1, -1)
    ymin, ymax = bbox[1], bbox[4]
    zmin, zmax = bbox[2], bbox[5]
    yaxis_refine_cutoff = ymax + (ymin - ymax) * y_proportion # the y-coordinate is inversed
    zaxis_refine_cutoff = zmin + (zmax - zmin) * x_proportion

    # obtain all points and their information
    all_geometric_points = gmsh.model.getEntities(0)
    points_to_refine, points_to_coarsen = utils.get_refine_coarse_points(all_geometric_points,
                                                                         yaxis_refine_cutoff, zaxis_refine_cutoff)

    # use gmsh setSize to set the characteristic length of nodes in refinement section
    if points_to_refine:
        gmsh.model.mesh.setSize(points_to_refine, LC_FINE)
        print(f"Sizes of {len(points_to_refine)} nodes has been set to {LC_FINE}")

    # use gmsh setSize to set the characteristic length of nodes in coarse section
    if points_to_coarsen:
        gmsh.model.mesh.setSize(points_to_coarsen, LC_COARSE)
        print(f"Sizes of {len(points_to_coarsen)} nodes has been set to {LC_COARSE}")

    MESH_FILE_NAME = "fine_elbow.msh"

print("Generating 3D mesh...")
gmsh.model.mesh.generate(3)
gmsh.write(MESH_FILE_NAME)
gmsh.finalize()

# ToDo: Needs to integrate the OpenFoam operations here (gmshToFoam, editing files, foamRun, foamToVTK)

# 3. transform results from VTK files (x and y are 8-dimensional: x,y,z,p,u,v,w,gradU)
file = utils.get_latest_vtk(VTK_FILE_PATTERN)
x_features, edge_index, y, pos, L_Char, raw_pos = utils.process_vtk_to_graph(file, MODE)

# 4. create Data object and save .pt file
graph_data = utils.create_tg_data(x_features, edge_index, y, pos)
graph_data.L_Char = torch.tensor(L_Char, dtype=torch.float)
torch.save(graph_data, f"{GRAPH_DATA_FILE_NAME}")
print(f"Saved graph data to file: {GRAPH_DATA_FILE_NAME}")

# 5. Model Training
if MODE.lower() == "train":
    data = torch.load(GRAPH_DATA_FILE_NAME, weights_only=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    print(f"Device selected: {device}")
    print(f"Dimensions of features: {data.x.shape[1]} (Expected 8)")
    print(f"Number of sample points: {data.x.shape[0]}")

    target = data.y[:, 4].view(-1, 1)

    # 切分训练/测试集
    num_nodes = data.x.shape[0]
    indices = torch.randperm(num_nodes)
    split = int(num_nodes * 0.8)
    train_idx = indices[:split]
    test_idx = indices[split:]

    model = MeshRefinementGNN(in_channels=data.x.shape[1], out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.SmoothL1Loss(beta=1.0)
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
            print(f"Epoch {epoch + 1:04d} | Train Loss (SmoothL1): {loss.item():.6f} | Test Loss: {test_loss.item():.6f}")

    if not os.path.exists(MODEL_SAVE_PATH):
        os.makedirs(MODEL_SAVE_PATH)
    full_save_path = os.path.join(MODEL_SAVE_PATH, MODEL_SAVE_NAME)
    torch.save(model.state_dict(), full_save_path)
    print(f"\nModel saved to: ./{full_save_path}")

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index)





