import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import os
import numpy as np

# --- 配置参数 ---
DATA_FILE = "ground_truth_graph.pt"
MODEL_SAVE_PATH = "gnn_model_elbow.pth"
LEARNING_RATE = 0.001
EPOCHS = 2000
HIDDEN_CHANNELS = 128


# ==========================================
# 1. 定义 GNN 模型 (GraphSAGE)
# ==========================================
class MeshRefinementGNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Convolution 1: input 8 dimensions -> 128 hidden dimensions
        self.conv1 = SAGEConv(in_channels, HIDDEN_CHANNELS)
        # Convolution 2: 128 hidden dimensions -> 128 hidden dimensions
        self.conv2 = SAGEConv(HIDDEN_CHANNELS, HIDDEN_CHANNELS)
        # Convolution 3: 128 hidden dimensions -> 128 hidden dimensions
        self.conv3 = SAGEConv(HIDDEN_CHANNELS, out_channels)
        # Layer 4: Output -> 1 (Log Size)
        self.conv4 = SAGEConv(HIDDEN_CHANNELS, out_channels)

    def forward(self, x, edge_index):
        # x: [N, 8], edge_index: [2, E]
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 3
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)

        # Layer 4 (Output)
        x = self.conv4(x, edge_index)

        return x