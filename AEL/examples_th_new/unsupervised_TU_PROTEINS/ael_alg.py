import torch
import torch.nn.functional as F

def custom(edge_index, x):
    shuffled_idx = torch.randperm(edge_index.size(1)).tolist()
    edge_index = edge_index[:,shuffled_idx]
    
    return edge_index,x