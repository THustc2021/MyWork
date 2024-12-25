import torch
import numpy as np
def drop_nodes(edge_index,x):
    edge_num, _ = edge_index.size()
    drop_num = int(edge_num * 0.2)
    idx_drop = np.random.choice(edge_num, drop_num, replace=False)
    idx_keep = [n for n in range(edge_num) if not n in idx_drop]
    edge_index = edge_index[:, idx_keep]

    return edge_index,x