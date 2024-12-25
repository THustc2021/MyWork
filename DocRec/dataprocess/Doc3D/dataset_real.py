import os
import random
from os.path import join as pjoin

import cv2
import numpy as np
from torch.utils.data import Dataset


class RealDataset(Dataset):

    def __init__(self, seg_path="/home/xtanghao/THPycharm/dataset/DocUnet Benchmark/crop",
                 grid_path="/home/xtanghao/THPycharm/dataset/DocUnet Benchmark/grid",
                 res_path="/home/xtanghao/THPycharm/dataset/DocUnet Benchmark/scan",
                 img_transform=None):
        super(RealDataset, self).__init__()

        data = []
        for p in os.listdir(seg_path):
            name = p[:-4]
            scan_name = name.split("_")[0]
            data.append((pjoin(seg_path, p), pjoin(grid_path, name+".npy"), pjoin(res_path,scan_name+".png")))

        random.shuffle(data)
        self.data = data
        self.img_transform = img_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        seg_path, grid_path, scan_path = self.data[item]

        img = cv2.cvtColor(cv2.imread(seg_path), cv2.COLOR_BGR2RGB)
        grid = np.load(grid_path)
        if grid.shape[0] == 2:
            grid = grid.transpose(1, 2, 0)
        scan_img = cv2.cvtColor(cv2.imread(scan_path), cv2.COLOR_BGR2RGB)

        if self.img_transform != None:
            res = self.img_transform(image=img, masks=[grid, scan_img])
            img = res["image"]
            grid, scan_img = res["masks"]

        return img / 255., grid, scan_img / 255.