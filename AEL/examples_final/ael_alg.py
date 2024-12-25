import torch
import numpy as np

def custom(edge_index, x):
    edge_index_shuffled = torch.randint(0, x.size(1), (2, x.size(1)))
    return edge_index,x