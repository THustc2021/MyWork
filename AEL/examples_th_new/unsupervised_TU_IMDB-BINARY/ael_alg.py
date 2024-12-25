import torch
import numpy as np

def custom(edge_index, x):
    
    node_num = x[0]
    _, edge_num = edge_index.size()
    sub_num = int(node_num * np.random.uniform(0.7, 0.9))
    
    idx_keep = np.random.choice(node_num, sub_num, replace=False)
    idx_dict = {idx_keep[n]:n for n in range(len(idx_keep)}
    
    x_new = x[idx_keep, :]
    
    mask = torch.tensor([idx in idx_keep for idx in edge_index.flatten()]).view(2, -1)
    edge_index_new = edge_index[:, mask.all(dim=0)]
    
    return edge_index,x