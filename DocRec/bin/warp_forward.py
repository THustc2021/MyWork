import os
import random

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A


if __name__ == '__main__':

    uv_dir = "/home/xtanghao/THPycharm/dataset/Doc3D/uv/1"
    uv_path = os.path.join(uv_dir, random.choice(os.listdir(uv_dir)))

    uv = cv2.imread(uv_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    msk = uv[:, :, 0]
    msk = cv2.resize(msk, (448*2, 448*2))
    img_dir = "/home/xtanghao/THPycharm/dataset/DIR300/gt"
    img_path = os.path.join(img_dir, random.choice(os.listdir(img_dir)))
    # img_path = uv_path.replace("uv", "geo_gt").replace("exr", "png")
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.
    # img = cv2.resize(img, (448*2, 448*2))
    img = A.RandomCrop(448*2, 448*2)(image=img)["image"]
    plt.imshow(img)
    plt.show()

    # 第一次变形
    uv_o_path = bin_utils.random_get_path(has_sub_dir=True)
    uv_o = cv2.imread(uv_o_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    uv_o = 2 * np.stack([uv_o[:, :, 2], 1 - uv_o[:, :, 1]], axis=2).astype(float) - 1
    uv_o = cv2.resize(uv_o, (448 * 2, 448 * 2))
    img = torch.from_numpy(img)[None].permute(0, 3, 1, 2)
    coords_ori = torch.stack(torch.meshgrid([torch.linspace(0, img.shape[3] - 1, img.shape[3]),
                                             torch.linspace(0, img.shape[2] - 1, img.shape[2])]))
    img_with_coords = torch.cat([img, coords_ori[None]], dim=1)
    img = F.grid_sample(img_with_coords, torch.from_numpy(uv_o)[None], align_corners=False)
    img_np = img.detach().cpu().numpy()[0, :3].transpose(1, 2, 0)
    plt.imshow(img_np)
    plt.show()

    uv = 2 * np.stack([uv[:, :, 2], 1 - uv[:, :, 1]], axis=2).astype(float) - 1
    uv = cv2.resize(uv, (448*2, 448*2))

    res = F.grid_sample(img, torch.from_numpy(uv)[None], align_corners=False)
    res_np = res.detach().cpu().numpy()[0, :3].transpose(1, 2, 0)
    plt.imshow(res_np)
    plt.show()

    bm_path = uv_path.replace("uv", "bm").replace("exr", "mat") # 这里有bug
    import hdf5storage as h5
    bm = h5.loadmat(bm_path)["bm"]
    # 归一化label到 [-1,1]
    label = bm.astype(float)
    label = label / np.array([label.shape[0], label.shape[1]])
    label = (label - 0.5*448/447) * 2
    label = cv2.resize(label, (448*2, 448*2))

    # 使用反向矫正
    res = F.grid_sample(res, torch.from_numpy(label)[None], align_corners=False)
    res_np = res.detach().cpu().numpy()[0].transpose(1, 2, 0)
    plt.imshow(res_np[:, :, :3])
    plt.show()

    # plt.imshow(res_np[:, :, 3])
    # plt.show()
    # plt.imshow(res_np[:, :, 4])
    # plt.show()
    res_final = F.grid_sample(res[:, :3], res[:, 3:].permute(0, 2, 3, 1), align_corners=False)
    res_final_np = res.detach().cpu().numpy()[0].transpose(1, 2, 0)
    plt.imshow(res_final_np[:, :, :3])
    plt.show()

    # 计算两者损失
    plt.imshow(np.abs(res_np[:, :, :3] - img))
    plt.show()

    # img_path = uv_path.replace("uv", "rembg").replace("exr", "png")
    # img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255.
    # # img = cv2.resize(img, (448, 448))
    # plt.imshow(img)
    # plt.show()