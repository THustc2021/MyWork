import os
import random
from copy import copy

import cv2
import hdf5storage as h5
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np

from utils.common import gen_ori_points
from utils.debug_utils import recoverImg
from utils.thinplatespline.batch import TPS

############# pre set
bs = 1
points_num = (4, 4)
radius = 10
save_dir = "figures"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# size = 3
# sn = size * size
# filter = torch.eye(sn, sn)
# filter[:, sn // 2] -= 1
# filter = filter.view(sn, size, size)
# filter = torch.cat([filter[:(sn // 2)], filter[(sn // 2 + 1):]], dim=0)[:, None]

random.seed(10)

################
image_path = "/home/xtanghao/THPycharm/dataset/Doc3D/rembg/4/101_5-pp_Page_433-v9c0001.png"
label_path = image_path.replace('rembg', 'bm').replace('png', 'mat')

img_np = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) # 不必转通道和归一化
bm = h5.loadmat(label_path)["bm"].astype(float)

cv2.imwrite(os.path.join(save_dir, "img_np.png"), img_np)

# 转tensor
img_t = torch.from_numpy(img_np)[None].permute(0, 3, 1, 2) / 255.
label = torch.from_numpy(bm)[None].float()
# 获得给定点
total_h, total_w = label.shape[1:3]
h, w = points_num
p_idx, p_ori = gen_ori_points((h, w), (total_h, total_w))
targ = label[torch.linspace(0, bs - 1, bs)[:, None].repeat(1, h * w).to(label).int(),
             [p_idx[0]] * bs,
             [p_idx[1]] * bs].transpose(1, 2).view(-1, 2, h, w)
# 转numpy
targ_np = targ[0].flatten(1).numpy().transpose(1, 0).astype(int)

# 原图中展示图像和控制点
zero_np = np.zeros_like(img_np)
for i in range(targ_np.shape[0]):
    img_np = cv2.circle(img_np, tuple(targ_np[i].tolist()), radius, (0, 0, 255), thickness=-1)
    zero_np = cv2.circle(zero_np, tuple(targ_np[i].tolist()), radius, (255, 0, 0), thickness=-1)
cv2.imwrite(os.path.join(save_dir, "ori_img_cptr.png"), img_np)
cv2.imwrite(os.path.join(save_dir, "zero_img_cptr.png"), zero_np)
# 展示目标控制点
p_ori_np = np.stack(np.meshgrid(np.linspace(0, img_np.shape[0]-1, points_num[0], dtype=int),
                       np.linspace(0, img_np.shape[1]-1, points_num[1], dtype=int)), axis=2).reshape(-1, 2)
zero_np = np.zeros_like(img_np)
for i in range(p_ori_np.shape[0]):
    zero_np = cv2.circle(zero_np, tuple(p_ori_np[i].tolist()), radius, (0, 0, 255), thickness=-1)
cv2.imwrite(os.path.join(save_dir, "zero_tgt_cptr.png"), zero_np)

#
# 获得完全TPS矫正流（归一化之后再算一次）
label = bm / np.array([bm.shape[0], bm.shape[1]])
label = (label - 0.5) * 2
label = torch.from_numpy(label)[None].float()
# 获得给定点
total_h, total_w = label.shape[1:3]
h, w = points_num
p_idx, p_ori = gen_ori_points((h, w), (total_h, total_w))
targ_old = label[torch.linspace(0, bs - 1, bs)[:, None].repeat(1, h * w).to(label).int(),
             [p_idx[0]] * bs,
             [p_idx[1]] * bs]
targ = targ_old.transpose(1, 2).view(-1, 2, h, w)
tps = TPS((total_h, total_w), device="cpu")
grid = tps(p_ori[None].float(), targ.flatten(2, 3).transpose(1, 2))
print(f"basic loss: {F.l1_loss(grid, label)}")      # 与未归一化label
recoverImg(img_t, grid.permute(0, 3, 1, 2))

# showCptr(img_t.float(), targ_old.float(), p_ori[None].float(), save=save_dir)

# # filter扫过label计算每个区域的局部变形程度
# coords = gen_coords(total_h, total_w)[None]
# fi = 0
# all_d = torch.zeros(bs, h * w)
#
# tgt_grid_x = label[..., 0][:, None] - coords[:, 0][:, None]
# tgt_grid_y = label[..., 1][:, None] - coords[:, 1][:, None]
# tgt_x_feat = F.conv2d(tgt_grid_x, filter, stride=1)
# tgt_y_feat = F.conv2d(tgt_grid_y, filter, stride=1)
# tgt_feat = torch.stack([tgt_x_feat, tgt_y_feat], dim=1).to(label)
#
# s = torch.std(tgt_feat, dim=2)
# d = torch.pow(s[:, 0], 2) + torch.pow(s[:, 1], 2)
# d = F.pad(d, (1, 1, 1, 1))
# d = d[torch.linspace(0, bs - 1, bs)[:, None].repeat(1, h * w).to(label).int(),
#       [p_idx[0]] * bs,
#       [p_idx[1]] * bs]
# all_d += d
#
# all_d = all_d.softmax(dim=1).view(bs, 1, h, w)
#
# # # 使用滑窗进行后处理，保留或去除滑窗内最大值
# # all_d_d, all_d_idx = F.max_pool2d_with_indices(all_d, kernel_size=(2, 2), return_indices=True)
# # all_d_n = F.max_unpool2d(all_d_d, indices=all_d_idx, kernel_size=(2, 2))
# # all_d = all_d - all_d_n
#
# # showImage(all_d)
# # showImage(img)
# # 保存变化图
# all_d_np = all_d[0, 0, 1:-1, 1:-1].numpy()
# all_d_np = (all_d_np - all_d_np.min()) / (all_d_np.max() -all_d_np.min())
# cv2.imwrite(os.path.join(save_dir, "deform map.png"), (all_d_np * 255).astype(np.uint8))
# #
# # k = all_d.numpy()[0, 0]
#
# k = all_d.clone()
# k[all_d < all_d.mean()] = 0
# grid = TPSByConf(label[0].permute(2, 0, 1), torch.cat([targ, k], dim=1)[0], threshold=1e-7, return_grid=True)
# print(f"basic loss: {F.l1_loss(grid, label)}")
# recoverImg_Tensor(img_t.float(), grid.permute(0, 3, 1, 2))
# # k = all_d_np
# # tr = k.mean() * 1.5
# # k[k<tr] = 0
# # k[k>0] = 1
# # k = np.pad(k, ((1, 1), (1, 1)), constant_values=1)
# # k = k.flatten()
# k[k > 0] = 1
# targ_np_u = targ_np * k.numpy()[0,0].transpose(1, 0).reshape(-1, 1).astype(int)
# zero_np = np.zeros_like(img_np)
# for i in range(p_ori_np.shape[0]):
#     zero_np = cv2.circle(zero_np, tuple(targ_np_u[i].tolist()), radius, (0, 0, 255), thickness=-1)
# cv2.imwrite(os.path.join(save_dir, "zero_ori_cptr_with_hdis.png"), zero_np)
# #
# # # 归一化label到 [-1,1]
# # grid = TPSByConf(label[0].permute(2, 0, 1).float(), torch.cat([targ, torch.from_numpy(k.reshape(*points_num))[None, None]], dim=1)[0], threshold=0, return_grid=True)
# # print(f"basic loss: {F.l1_loss(grid, label)}")
# # recoverImg_Tensor(img_t.float(), grid.permute(0, 3, 1, 2))
# # #
# # l = all_d.clone()
# # l[all_d > all_d.mean()] = 0
# # grid = TPSByConf(label[0].permute(2, 0, 1), torch.cat([targ, l], dim=1)[0], threshold=1e-7, return_grid=True)
# # print(f"basic loss: {F.l1_loss(grid, label)}")
# # recoverImg_Tensor(img.float(), grid.permute(0, 3, 1, 2))