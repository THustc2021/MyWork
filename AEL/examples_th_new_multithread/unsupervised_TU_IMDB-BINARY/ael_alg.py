import random
    import numpy as np

    node_num = np.max(edge_index) + 1
    drop_num = int(node_num * 0.2)
    
    idx_drop = random.sample(range(node_num), drop_num)
    idx_keep = list(set(range(node_num)) - set(idx_drop))
    
    edge_list = edge_index.T.tolist()
    updated_edge_list = [edge for edge in edge_list if edge[0] in idx_keep and edge[1] in idx_keep]
    
    updated_edge_index = np.array(updated_edge_list).T
    
    return edge_index,x