import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv


class MeshOptimizerGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # Encoder: 将物理/几何特征映射到高维潜空间
        self.node_encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU()
        )

        # Processor: 多层消息传递 (Message Passing)
        # 这里可以使用 EdgeConv 或 GAT，捕捉局部拓扑结构
        self.layers = nn.ModuleList()
        for _ in range(5):
            self.layers.append(
                EdgeConv(nn.Sequential(
                    nn.Linear(2 * hidden_channels, hidden_channels),
                    nn.ReLU(),
                    nn.Linear(hidden_channels, hidden_channels)
                ))
            )

        # Decoder: 将潜特征映射回物理空间的位移
        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)  # Output: (dx, dy, dz)
        )

    def forward(self, x, edge_index):
        # x: [num_nodes, in_features]
        # edge_index: [2, num_edges]

        # 1. Encoding
        h = self.node_encoder(x)

        # 2. Message Passing
        for layer in self.layers:
            h_res = layer(h, edge_index)
            h = h + h_res  # Residual connection 防止梯度消失

        # 3. Decoding
        displacement = self.decoder(h)

        # 4. Hard Constraint: 强制边界节点不移动
        # mask = x[:, is_boundary_idx] == 0
        # displacement = displacement * mask

        return displacement