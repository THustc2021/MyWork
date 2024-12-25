import os

import cv2
import h5py as h5
import numpy as np

seg_dir = "/home/xtanghao/THPycharm/dataset/UVDoc_final/seg"
save_dir = "/home/xtanghao/THPycharm/dataset/UVDoc_final/msk"

# os.mkdir(save_dir)
for p in os.listdir(seg_dir):
    seg_path = os.path.join(seg_dir, p)
    with h5.File(seg_path, "r") as file:
        mask = np.array(file["seg"][:].T)
    cv2.imwrite(os.path.join(save_dir, p.replace("mat", "png")), mask * 255)
    print(f"write {p} done.")