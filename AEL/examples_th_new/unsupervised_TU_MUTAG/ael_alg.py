import torch
import numpy as np

def custom(edge_index, x):

    node_num, _ = x.size()
    _, edge_num = edge_index.size()

    drop_percentage = 0.2
    drop_num = int(edge_num * drop_percentage)

    idx_drop = np.random.choice(edge_num, drop_num, replace=False)
    idx_keep = [n for n in range(edge_num) if not n in idx_drop]

    edge_index = edge_index[:, idx_keep]

    return edge_index,x