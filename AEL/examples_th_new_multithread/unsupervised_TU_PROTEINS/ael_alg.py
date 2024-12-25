import torch

def custom(edge_index, x):
    edge_num = edge_index.size(1)
    mask_prob = 0.2
    
    random_mask = torch.rand(edge_num) < mask_prob

    edge_index = edge_index[:, ~random_mask]

    return edge_index,x