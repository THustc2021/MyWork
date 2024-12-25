import os
import random
import numpy as np
import cv2
import json
import h5py as h5
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from os.path import join as pjoin

from dataprocess.UVDoc.data_utils import crop_image_tight


class UVDoc_All_Dataset(Dataset):

    def __init__(self, root_dir, img_transform=None):
        data = []
        for namex in os.listdir(os.path.join(root_dir, "img")):
            if namex.endswith(".png"):
                name = namex[:-4]   # 获取名字
            else:
                continue
            with open(pjoin(root_dir, "metadata_sample", f"{name}.json"), "r") as f:
                file = json.load(f)
                sample_name = file["geom_name"]
                t_name = file["texture_name"]
            img_path = pjoin(root_dir, "img", f"{name}.png")
            alb_path = pjoin(root_dir, "warped_textures", f"{name}.png")
            seg_path = pjoin(root_dir, "seg", f"{sample_name}.mat")
            grid2D_path = pjoin(root_dir, "grid2d", f"{sample_name}.mat")
            grid3D_path = pjoin(root_dir, "grid3d", f"{sample_name}.mat")
            texture_path = pjoin(root_dir, "textures", f"{t_name.split('/')[-1]}")

            data.append((img_path, seg_path, alb_path, grid2D_path, grid3D_path, texture_path))
        random.shuffle(data)

        self.data = data
        self.img_transform = img_transform

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.data)

    def crop_tight(self, img_RGB, grid2D):
        # The incoming grid2D array is expressed in pixel coordinates (resolution of img_RGB before crop/resize)
        size = img_RGB.shape
        img, top, bot, left, right = crop_image_tight(img_RGB, grid2D)
        img = cv2.resize(img, self.img_size)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()

        grid2D[0, :, :] = (grid2D[0, :, :] - left) / (size[1] - left - right)
        grid2D[1, :, :] = (grid2D[1, :, :] - top) / (size[0] - top - bot)
        grid2D = (grid2D * 2.0) - 1.0

        return img, grid2D

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        distorted_image_path = self.data[index][0]
        seg_path = self.data[index][1]
        alb_image_path = self.data[index][2]
        bm_path = self.data[index][3]
        texture_path = self.data[index][5]

        distorted_image = cv2.cvtColor(cv2.imread(distorted_image_path), cv2.COLOR_BGR2RGB)
        alb_image = cv2.cvtColor(cv2.imread(alb_image_path), cv2.COLOR_BGR2RGB)
        texture_image = cv2.cvtColor(cv2.imread(texture_path), cv2.COLOR_BGR2RGB)

        with h5.File(bm_path, "r") as file:
            bm = np.array(file["grid2d"][:].T)
        # 归一化label到 [-1,1]
        bm = bm / np.array([bm.shape[0], bm.shape[1]])
        bm = (bm - 0.5) * 2
        with h5.File(seg_path, "r") as file:
            mask = np.array(file["seg"][:].T)

        distorted_image = distorted_image * mask[..., None]
        if self.img_transform != None:
            res = self.img_transform(image=distorted_image, masks=[alb_image, texture_image])
            distorted_image = res["image"]
            alb_image, texture_image = res["masks"]
            # 获得分辨率合适的bm
            bm = torch.from_numpy(bm.transpose(2, 0, 1))
            bm = F.interpolate(
                bm[None], size=distorted_image.shape[1:], mode="bilinear", align_corners=True
            )[0]

        return distorted_image / 255., alb_image / 255., bm, texture_image / 255.


if __name__ == '__main__':

    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    d = UVDoc_All_Dataset("/home/xtanghao/THPycharm/dataset/UVDoc_final", img_transform=ToTensorV2(transpose_mask=True))
    d[0]