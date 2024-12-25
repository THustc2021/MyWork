import torch
import numpy as np
def mask_nodes(data):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * 0.2)

    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = torch.tensor(np.random.normal(loc=0.5, scale=0.5, size=(mask_num, feat_dim)), dtype=torch.float32)

    return data
