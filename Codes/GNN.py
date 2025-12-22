import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class GNN(MessagePassing):
    def __init__(self, in_channels, out_channels, num_layers, hidden_layer_dim=32,
                 dropout=0.1, **kwargs):
        super(GNN, self).__init__(aggr='mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.hidden_layer_dim = hidden_layer_dim
        self.dropout = nn.Dropout(dropout)

        # message passing for neighbours
        self.message_mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_layer_dim), # linear is generally used
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, hidden_layer_dim)
        )

        # update nodes
        self.update_mlp = nn.Sequential(
            nn.Linear(in_channels + hidden_layer_dim, hidden_layer_dim),
            nn.ReLU(),
            nn.Linear(hidden_layer_dim, out_channels)
        )

        # prevent overfitting
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        concatenate = torch.cat([x_i, x_j], dim=1)
        return self.message_mlp(concatenate)

    def update(self, aggr_out, x):
        concatenate = torch.cat([x, aggr_out], dim=1)
        return self.update_mlp(concatenate)