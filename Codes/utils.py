import torch
from torch_geometric.data import Data

def create_tg_data(x, edge_index):
    coordinate_tensor = torch.tensor(x, dtype=torch.float)
    graph_data = Data(x=coordinate_tensor, edge_index=edge_index)
    return graph_data
