import os
import random

import albumentations.pytorch
import cv2
import torch
import torch.nn.functional as F
import hdf5storage as h5
import numpy as np
from os.path import join as pjoin
from torch.utils.data import Dataset

from utils.thinplatespline.batch import TPS


class TPSDataset(Dataset):

    def __init__(self, root_dirs, bm_dir="/home/xtanghao/THPycharm/dataset/Doc3D/bm",
                 num_points_per_axis=3, img_transform=None):
        super(TPSDataset, self).__init__()

        data = []
        for d in root_dirs:
            for dd in os.listdir(d):
                if os.path.isdir(pjoin(d, dd)):
                    data.extend([pjoin(d, dd, ddd) for ddd in os.listdir(pjoin(d, dd))])
                else:
                    data.append(pjoin(d, dd))
        random.shuffle(data)
        self.data = data

        bm_data = []
        for d in os.listdir(bm_dir):
            bm_data.extend([pjoin(bm_dir, d, dd) for dd in os.listdir(pjoin(bm_dir, d))])
        random.shuffle(bm_data)
        self.bm_data = bm_data

        self.num_points_per_axis = num_points_per_axis
        self.img_transform = img_transform

        # 依据bm图生成点
        H = W = 448
        ys = np.linspace(0, H - 1, self.num_points_per_axis, dtype=int)
        ys = ys.repeat(self.num_points_per_axis)
        xs = np.linspace(0, W - 1, self.num_points_per_axis, dtype=int)
        xs = xs[:, None].repeat(self.num_points_per_axis, axis=1).transpose().flatten()
        points_ori = (list(xs), list(ys))
        self.points_ori = points_ori
        Y = torch.tensor(self.points_ori).transpose(0, 1).float()
        Y[..., 0] = 2 * Y[..., 0] / (W - 1) - 1
        Y[..., 1] = 2 * Y[..., 1] / (H - 1) - 1
        self.points_ori_t_norm = Y

    def __len__(self):
        return len(self.data)

    def _apply_tps(self, img_t):

        # 加载所有控制点
        bm = h5.loadmat(random.choice(self.bm_data))["bm"]
        # bm = h5.loadmat("/home/xtanghao/THPycharm/dataset/Doc3D/bm/12/135_6-ns_Page_648-M4w0001.mat")["bm"]
        # 归一化bm到 [-1,1]
        bm = bm.astype(float)
        bm = bm / np.array([bm.shape[0], bm.shape[1]])
        bm = (bm - 0.5) * 2

        # 找到若干个控制点
          # ([x1, ...], [y1, ...])
        points = bm[self.points_ori[::-1]]

        X = torch.from_numpy(points).to(img_t.device).float()
        Y = self.points_ori_t_norm.to(img_t.device)

        # 进行校正
        tpsb = TPS(size=img_t.shape[1:], device=img_t.device)
        warped_grid_b = tpsb(X[None, ...], Y[None, ...]).to(img_t.device)  # tpsb(target, source)

        # 变形处理
        ten_wrp_b = torch.grid_sampler_2d(img_t[None, ...],
            warped_grid_b,
            0, 0, False)

        return ten_wrp_b[0], X

    def _apply_uv(self, img_t):

        # 加载所有控制点
        bm_o_path = random.choice(self.bm_data)
        uv_o_path = pjoin(os.path.dirname(bm_o_path).replace("bm", "uv"), os.path.basename(bm_o_path).replace("mat", "exr"))
        while not os.path.exists(uv_o_path):
            bm_o_path = random.choice(self.bm_data)
            uv_o_path = pjoin(os.path.dirname(bm_o_path).replace("bm", "uv"),
                              os.path.basename(bm_o_path).replace("mat", "exr"))

        bm = h5.loadmat(bm_o_path)["bm"]
        # bm = h5.loadmat("/home/xtanghao/THPycharm/dataset/Doc3D/bm/12/135_6-ns_Page_648-M4w0001.mat")["bm"]
        # 归一化bm到 [-1,1]
        bm = bm.astype(float)
        bm = bm / np.array([bm.shape[0], bm.shape[1]])
        bm = (bm - 0.5) * 2
        points = bm[self.points_ori[::-1]]
        X = torch.from_numpy(points).to(img_t.device).float()

        uv_o = cv2.imread(uv_o_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        msk = torch.from_numpy(uv_o[..., 0])[None, None]
        uv_o = 2 * np.stack([uv_o[:, :, 2], 1 - uv_o[:, :, 1]], axis=2).astype(float) - 1

        img = F.grid_sample(img_t[None], torch.from_numpy(uv_o)[None].float(), align_corners=False) * msk
        return img[0], X

    def __getitem__(self, item):

        img_path = self.data[item]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGBA2RGB)

        if self.img_transform != None:
            img = self.img_transform(image=img)["image"]

        img = albumentations.pytorch.ToTensorV2(transpose_mask=True)(image=img)["image"] / 255.

        img_wrp, x = self._apply_uv(img)

        return img_wrp, x

if __name__ == '__main__':

    TPSDataset(["/home/xtanghao/THPycharm/dataset/docs-sm"], img_transform=albumentations.RandomCrop(448, 448))[0]