import torch
import numpy as np
def drop_nodes(edge_index,x):
    node_num, _ = x.size()
    _, edge_num = edge_index.size()
    drop_num = int(node_num * 0.2)
    idx_drop = np.random.choice(node_num, drop_num, replace=False)
    idx_nondrop = [n for n in range(node_num) if not n in idx_drop]
    idx_dict = {idx_nondrop[n]:n for n in list(range(node_num - drop_num))}
    edge_index = edge_index.numpy()
    adj = torch.zeros((node_num, node_num))
    adj[edge_index[0], edge_index[1]] = 1
    adj[idx_drop, :] = 0
    adj[:, idx_drop] = 0
    edge_index = adj.nonzero().t()

    return edge_index,x