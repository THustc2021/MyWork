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

from models.UnifyNet import UnifyNet
from utils.common import reload_model


def rec_geo(opt, img_size=(896, 896)):
    # img_size: (w, h)
    inp_dir = opt.inp_path
    seg_dir = opt.seg_path
    out_dir = opt.out_path
    device = opt.device
    show_not_save = opt.show_not_save

    if not show_not_save and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # seg_model = DeepLabv3()
    # reload_model(seg_model, opt.seg_model_path, not_use_parrel_trained=True)
    # seg_model.to(device)
    # seg_model.eval()

    model = UnifyNet()
    model.load_model(opt.model_path, not_use_parrel_trained=True)
    model.to(device)
    model.eval()
    # ill_model = IllRec()
    # ill_model.load_model(opt.ill_model_path)
    # ill_model.to(device)
    # ill_model.eval()
    #
    # geo_model = GeoRec()
    # geo_model.load_model(opt.geo_model_path)
    # geo_model.to(device)
    # geo_model.eval()

    for name in os.listdir(inp_dir):
        p = os.path.join(inp_dir, name)

        # 读取图像
        img_ori = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        img_msk = cv2.cvtColor(cv2.imread(os.path.join(seg_dir, name)), cv2.COLOR_BGR2RGB)

        h, w, _ = img_ori.shape
        if h % 16 != 0:   # 根据原始大小来进行处理（可能显存不足）
            h += 16 - h % 16
        if w % 16 != 0:
            w += 16 - w % 16
        ori_img_size = (w, h)
        print(ori_img_size)

        # 调整大小
        ori_img_t = cv2.resize(img_ori, ori_img_size)
        img_msk = cv2.resize(img_msk, img_size)
        # 调整通道顺序
        ori_img_t = torch.from_numpy(img_msk).permute(2, 0, 1)[None].to(device) / 255.
        img_t = torch.from_numpy(img_msk).permute(2, 0, 1)[None].to(device) / 255.
        # 模型处理
        with torch.no_grad():
            # msk = seg_model(img_tensor)
            # rembg_tensor = (msk > 0).float() * img_tensor
            output = model(img_t, return_final_result=True)
            # output = model(rembg_tensor, return_final_result=True)
            # output_ill = ill_model(ori_img_t)   # 原始分辨率
            # output_geo = geo_model(rembg_tensor)    # 裁剪分辨率
            # grid = F.interpolate(output_geo, (h, w), mode="bilinear")
            # 重采样
            # output = F.grid_sample(output_ill, grid.permute(0, 2, 3, 1), align_corners=False)
        # 结果
        output = output.detach().cpu()[0] * 255.
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
    parser.add_argument('--inp_path', default='/home/xtanghao/THPycharm/dataset/DocUnet Benchmark/crop')
    parser.add_argument('--seg_path',
                        default='/home/xtanghao/THPycharm/dataset/DocUnet Benchmark/seg/final')
    parser.add_argument('--out_path',  default='/home/xtanghao/THPycharm/dataset/DocUnet Benchmark/results/ZZ_Final_geo_and_ill')
    parser.add_argument('--model_path',
                        default="/home/xtanghao/DocRec/results/UnifyNet/Final_geo_and_ill/model_0.pth")
    parser.add_argument('--show_not_save', default=False, help="whether save or show")
    parser.add_argument('--device', default="cuda")

    opt = parser.parse_args()

    rec_geo(opt)


if __name__ == '__main__':
    main()