import copy
import numpy as np
import matplotlib.pylab as plt
from torch_geometric.utils.convert import to_networkx
import networkx as nx
import matplotlib.patches as mpatches
from matplotlib import font_manager
import pickle
from typing import Callable, List, Optional
import os
import json
import torch

from aug import TUDataset_aug as TUDataset
from torch_geometric.data import Data, InMemoryDataset, download_url

def plot_nx(ax, G, node_attr, edge_attr, head_index, pos=None):
    nodelist = G.nodes()

    ax.axis('off')

    node_color = ['#32CD32' if i else '#1f78b4' for i in node_attr]
    node_color[head_index] = 'red'
    nx.draw_networkx_nodes(G, pos,
                           nodelist=nodelist,
                           node_size=150,
                           node_color=node_color,
                           ax=ax
                           )
    nx.draw_networkx_edges(G, pos,
                           edgelist=G.edges,
                           width=[3 if i else 1 for i in edge_attr],
                           edge_color='black',
                           style=['-' if i else '--' for i in edge_attr],
                           ax=ax
                           )
    nx.draw_networkx_edge_labels(G, pos, dict(zip(G.edges, range(len(G.edges)))), ax=ax)

# 参数设置
DS = "NCI1"
aug = None
eva = None

path = './calibrili.ttf'
font_manager.fontManager.addfont(path)
plt.rcParams['font.family'] = 'Calibri'

path = "."

fig, ax_dict = plt.subplot_mosaic([['A', 'B', 'C', 'D', 'E']],empty_sentinel="BLANK")
fig.subplots_adjust(wspace = 0.15,hspace=0.15)

###

data = TUDataset(path, name=DS, aug=aug, eva=eva)
# 选中图并处理结点
_, p_graph_aug = data[5]
print(p_graph_aug)
node_attr = np.zeros(p_graph_aug.x.shape[0])
node_attr[0] = 1
edge_attr = np.zeros(p_graph_aug.edge_index.shape[1])
edge_attr[20:] = 1
# 转无向图
G = to_networkx(p_graph_aug).to_undirected()
pos = nx.kamada_kawai_layout(G)
##########
plot_nx(ax_dict['A'], G, node_attr, edge_attr,20,pos=pos)
ax_dict['A'].set_title('No Aug',fontsize = 17)
ax_dict['A'].set_aspect('equal')

##########
aug_point_path = f"/home/xtanghao/THPycharm/AEL_main/examples/unsupervised_TU_{DS}/ael_seeds/seeds.json"
with open(aug_point_path, 'r', encoding='utf-8') as file:
    # 使用json.load()方法解析JSON数据
    ael_code = json.load(file)[0]["code"]
with open("./ael_alg.py", "w") as file:
    # Write the code to the file
    file.write(ael_code)
from ael_alg import drop_nodes as eva

data = TUDataset(path, name=DS, aug="dnodes", eva=eva)
# 选中图并处理结点
_, data_aug = data[5]
node_num, _ = data.x.size()
edge_idx = data_aug.edge_index.numpy()
_, edge_num = edge_idx.shape
idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]  # 找到所有与边关联的结点，即非缺失结点

node_num_aug = len(idx_not_missing)  # 计算非缺失结点的数量
data_aug.x = data_aug.x[idx_not_missing]  # 仅保留非缺失结点的特征而不是所有的（更新增强数据的特征）
idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}  # 将非缺失结点的索引映射到新的索引
edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
            not edge_idx[0, n] == edge_idx[1, n]]  # 使用新的结点索引更新边的索引
data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
print(data_aug)
node_attr = np.zeros(data_aug.x.shape[0])
node_attr[0] = 1
edge_attr = np.zeros(data_aug.edge_index.shape[1])
edge_attr[20:] = 1
# 转无向图
G = to_networkx(data_aug).to_undirected()
pos = nx.kamada_kawai_layout(G)

plot_nx(ax_dict['B'],G,node_attr,edge_attr,20,pos=pos)
ax_dict['B'].set_title('Aug',fontsize = 17)
ax_dict['B'].set_aspect('equal')
####

aug_point_path = f"/home/xtanghao/THPycharm/AEL_main/examples/unsupervised_TU_{DS}/th_llm/ael_results/pops/population_generation_0.json"
with open(aug_point_path, 'r', encoding='utf-8') as file:
    # 使用json.load()方法解析JSON数据
    ael_code = json.load(file)[0]["code"]
with open("./ael_alg.py", "w") as file:
    # Write the code to the file
    file.write(ael_code)
from ael_alg import drop_nodes as eva

data = TUDataset(path, name=DS, aug="dnodes", eva=eva)
# 选中图并处理结点
_, data_aug = data[5]
node_num, _ = data.x.size()
edge_idx = data_aug.edge_index.numpy()
_, edge_num = edge_idx.shape
idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]  # 找到所有与边关联的结点，即非缺失结点

node_num_aug = len(idx_not_missing)  # 计算非缺失结点的数量
data_aug.x = data_aug.x[idx_not_missing]  # 仅保留非缺失结点的特征而不是所有的（更新增强数据的特征）

idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}  # 将非缺失结点的索引映射到新的索引
edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
            not edge_idx[0, n] == edge_idx[1, n]]  # 使用新的结点索引更新边的索引
data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
print(data_aug)
node_attr = np.zeros(data_aug.x.shape[0])
node_attr[0] = 1
edge_attr = np.zeros(data_aug.edge_index.shape[1])
edge_attr[20:] = 1
# 转无向图
G = to_networkx(data_aug).to_undirected()
pos = nx.kamada_kawai_layout(G)

plot_nx(ax_dict['C'],G,node_attr,edge_attr,20,pos=pos)
ax_dict['C'].set_title('laug',fontsize = 17)
ax_dict['C'].set_aspect('equal')

#######

aug_point_path = f"/home/xtanghao/THPycharm/AEL_main/examples/unsupervised_TU_{DS}/ael_results/pops/population_generation_1.json"
with open(aug_point_path, 'r', encoding='utf-8') as file:
    # 使用json.load()方法解析JSON数据
    ael_code = json.load(file)[0]["code"]
with open("./ael_alg.py", "w") as file:
    # Write the code to the file
    file.write(ael_code)
from ael_alg import drop_nodes as eva

data = TUDataset(path, name=DS, aug="dnodes", eva=eva)
# 选中图并处理结点
_, p_graph_aug = data[5]
node_num, _ = data.x.size()
edge_idx = data_aug.edge_index.numpy()
_, edge_num = edge_idx.shape
idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]  # 找到所有与边关联的结点，即非缺失结点

node_num_aug = len(idx_not_missing)  # 计算非缺失结点的数量
data_aug.x = data_aug.x[idx_not_missing]  # 仅保留非缺失结点的特征而不是所有的（更新增强数据的特征）

idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}  # 将非缺失结点的索引映射到新的索引
edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
            not edge_idx[0, n] == edge_idx[1, n]]  # 使用新的结点索引更新边的索引
data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
print(p_graph_aug)
node_attr = np.zeros(p_graph_aug.x.shape[0])
node_attr[0] = 1
edge_attr = np.zeros(p_graph_aug.edge_index.shape[1])
edge_attr[20:] = 1
# 转无向图
G = to_networkx(p_graph_aug).to_undirected()
pos = nx.kamada_kawai_layout(G)

plot_nx(ax_dict['D'],G,node_attr,edge_attr,20,pos=pos)
ax_dict['D'].set_title('laug_evo',fontsize = 17)
ax_dict['D'].set_aspect('equal')
##########

aug_point_path = f"/home/xtanghao/THPycharm/AEL_main/examples/unsupervised_TU_{DS}/ael_results/pops_best/population_generation_1.json"
with open(aug_point_path, 'r', encoding='utf-8') as file:
    # 使用json.load()方法解析JSON数据
    ael_code = json.load(file)["code"]
with open("./ael_alg.py", "w") as file:
    # Write the code to the file
    file.write(ael_code)
from ael_alg import drop_nodes as eva

data = TUDataset(path, name=DS, aug="dnodes", eva=eva)
# 选中图并处理结点
_, p_graph_aug = data[5]
node_num, _ = data.x.size()
edge_idx = data_aug.edge_index.numpy()
_, edge_num = edge_idx.shape
idx_not_missing = [n for n in range(node_num) if (n in edge_idx[0] or n in edge_idx[1])]  # 找到所有与边关联的结点，即非缺失结点

node_num_aug = len(idx_not_missing)  # 计算非缺失结点的数量
data_aug.x = data_aug.x[idx_not_missing]  # 仅保留非缺失结点的特征而不是所有的（更新增强数据的特征）

idx_dict = {idx_not_missing[n]: n for n in range(node_num_aug)}  # 将非缺失结点的索引映射到新的索引
edge_idx = [[idx_dict[edge_idx[0, n]], idx_dict[edge_idx[1, n]]] for n in range(edge_num) if
            not edge_idx[0, n] == edge_idx[1, n]]  # 使用新的结点索引更新边的索引
data_aug.edge_index = torch.tensor(edge_idx).transpose_(0, 1)
print(p_graph_aug)
node_attr = np.zeros(p_graph_aug.x.shape[0])
node_attr[0] = 1
edge_attr = np.zeros(p_graph_aug.edge_index.shape[1])
edge_attr[20:] = 1
# 转无向图
G = to_networkx(p_graph_aug).to_undirected()
pos = nx.kamada_kawai_layout(G)

plot_nx(ax_dict['E'],G,node_attr,edge_attr,20,pos=pos)
ax_dict['E'].set_title('laug_evo_eva',fontsize = 17)
ax_dict['E'].set_aspect('equal')

#######
plt.tight_layout()
fig.subplots_adjust(wspace=0.3)
# plt.savefig('ba2.svg',format='svg', dpi=1200)
plt.show()
