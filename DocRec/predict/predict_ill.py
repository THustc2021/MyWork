import os
import random

import matplotlib.pyplot as plt

import cv2
import torch
import numpy as np
import hdf5storage as h5
import torch.nn.functional as F

import argparse

from PIL import Image

from models.IllRec.ill_decoderv7 import IllRec
from models.UnifyNet import UnifyNet
from utils.common import reload_model


def rec_geo(opt, img_size=(896, 896)):
    # img_size: (w, h)
    inp_dir = opt.inp_path
    out_dir = opt.out_path
    device = opt.device
    show_not_save = opt.show_not_save

    if not show_not_save and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ill_model = IllRec()
    ill_model.load_model(opt.model_path, not_use_parrel_trained=False)
    ill_model.to(device)
    ill_model.eval()
    #
    # geo_model = GeoRec()
    # geo_model.load_model(opt.geo_model_path)
    # geo_model.to(device)
    # geo_model.eval()

    for name in os.listdir(inp_dir):
        p = os.path.join(inp_dir, name)

        # 读取图像
        img_ori = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)

        h, w, _ = img_ori.shape
        if h % 16 != 0:   # 根据原始大小来进行处理（可能显存不足）
            h += 16 - h % 16
        if w % 16 != 0:
            w += 16 - w % 16
        ori_img_size = (w, h)
        print(ori_img_size)

        # 调整大小
        ori_img = cv2.resize(img_ori, img_size)
        # 调整通道顺序
        ori_img_t = torch.from_numpy(ori_img).permute(2, 0, 1)[None].to(device) / 255.
        # 模型处理
        with torch.no_grad():

            output_ill = ill_model(ori_img_t)   # 原始分辨率

        # 结果
        output = output_ill.detach().cpu()[0] * 255.
        resImg = output.numpy().transpose(1, 2, 0).astype(np.uint8)
        # save or show
        resImg = cv2.resize(resImg, (w, h))
        # resImg = postProcess(resImg)
        resImg = Image.fromarray(resImg)
        if not show_not_save:
            resImg.save(os.path.join(out_dir, name))
        else:
            plt.figure()
            plt.subplot(1, 2, 1)
            plt.imshow(img_ori)
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.imshow(resImg)
            plt.axis('off')
            plt.show()

        print(f"{p} done.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inp_path', default='/home/xtanghao/THPycharm/dataset/DocUnet Benchmark/results/ZZ_Geo_noback')
    parser.add_argument('--out_path',  default='/home/xtanghao/THPycharm/dataset/DocUnet Benchmark/results/ZZ_Ill_noback')
    parser.add_argument('--model_path',
                        default="/home/xtanghao/DocRec/results/UnifyNet/trained_in_both/model_4.pth")
    parser.add_argument('--show_not_save', default=False, help="whether save or show")
    parser.add_argument('--device', default="cuda")

    opt = parser.parse_args()

    rec_geo(opt)


if __name__ == '__main__':
    main()