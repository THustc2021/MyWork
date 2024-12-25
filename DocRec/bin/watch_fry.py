import cv2
import torch

img1 = cv2.cvtColor(cv2.imread("/home/xtanghao/THPycharm/dataset/diw5k/rembg/07oskB_4.png"), cv2.COLOR_BGR2RGB) / 255.
img2 = cv2.cvtColor(cv2.imread("/home/xtanghao/THPycharm/dataset/diw5k/rembg/0otaOu_4.png"), cv2.COLOR_BGR2RGB) / 255.
seg_mask_img2 = cv2.imread("/home/xtanghao/THPycharm/dataset/diw5k/msk/0otaOu_4.png", cv2.IMREAD_UNCHANGED) // 255

img1_t = torch.from_numpy(img1).permute(2, 0, 1)
img2_t = torch.from_numpy(img2).permute(2, 0, 1)

C, H, W = img1_t.shape
beta = 0.03

# 主要图像
fre1 = torch.fft.fft2(img1_t, dim=(1, 2))  # 变换得到的频域图数据是复数组成的
fre1 = torch.fft.fftshift(fre1, dim=[1, 2])

fre2 = torch.fft.fft2(img2_t, dim=(1, 2))
fre2 = torch.fft.fftshift(fre2, dim=[1, 2])

# 获得空白背景的图像
# blank = torch.ones_like(img_t)
blank = torch.from_numpy(seg_mask_img2)[None]
blank_fre = torch.fft.fft2(blank, dim=(1, 2))
blank_fre = torch.fft.fftshift(blank_fre, dim=(1, 2))

# 替换低频信号
# fre[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)] = \
#     blank_fre[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)]
# fre[:, :int(beta * H), :int(beta * W)] = blank_fre[:, :int(beta * H), :int(beta * W)]
fre11 = fre1.clone()
fre3 = fre2.clone()
fre1[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)] = \
    fre2[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)]
fre2[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)] = \
    blank_fre[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)]
fre3[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)] = \
    fre11[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)]

# 逆变换还原
fre1 = torch.fft.ifftshift(fre1, dim=[1, 2])  # 移动到原来位置
img_onlyphase1 = torch.abs(torch.fft.ifft2(fre1, dim=(1, 2)) ) # 还原为空间域图像
fre2 = torch.fft.ifftshift(fre2, dim=[1, 2])  # 移动到原来位置
img_onlyphase2 = torch.abs(torch.fft.ifft2(fre2, dim=(1, 2)) ) # 还原为空间域图像
fre3 = torch.fft.ifftshift(fre3, dim=[1, 2])  # 移动到原来位置
img_onlyphase3 = torch.abs(torch.fft.ifft2(fre3, dim=(1, 2)) ) # 还原为空间域图像

from utils.debug_utils import *
showTensorImg(img1_t[None])
showTensorImg(img2_t[None])
showTensorImg(blank[None], cmap="gray")
showTensorImg(img_onlyphase1[None])
showTensorImg(img_onlyphase2[None])
showTensorImg(img_onlyphase3[None])