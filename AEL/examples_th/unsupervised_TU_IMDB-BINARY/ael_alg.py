import numpy as np

def custom(edge_index, x):
    node_num = np.max(edge_index) + 1
    drop_percentage = 0.3
    drop_num = int(node_num * drop_percentage)

    edge_index_copy = np.copy(edge_index)
    np.random.shuffle(edge_index_copy.T)
    edge_index_copy = edge_index_copy[:, drop_num:]

    return edge_index,x