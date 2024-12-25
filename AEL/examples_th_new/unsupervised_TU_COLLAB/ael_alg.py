import torch
import numpy as np

def custom(edge_index, x):
    # Randomly select a subset of edges to update their attributes
    edge_num = edge_index.size(1)
    mask_num = int(edge_num * 0.1)
    idx_mask = np.random.choice(edge_num, mask_num, replace=False)
    
    selected_edges = x[edge_index[:, idx_mask][0]] + x[edge_index[:, idx_mask][1]]
    selected_edges *= torch.tensor(np.random.uniform(-1, 1, (mask_num, x.size(1))), dtype=torch.float32)

    return edge_index,x