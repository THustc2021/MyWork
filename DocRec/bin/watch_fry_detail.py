import cv2
import torch
from utils.debug_utils import *

img = cv2.cvtColor(cv2.imread("/home/xtanghao/THPycharm/dataset/DocUnet Benchmark/results/GeoTr_geo/41_1 copy_geo.png"), cv2.COLOR_BGR2RGB) / 255.
# img = cv2.cvtColor(cv2.imread("/home/xtanghao/THPycharm/dataset/diw5k/img/0TLBj6_4.png"), cv2.COLOR_BGR2RGB) / 255.
img = cv2.resize(img, (448, 448))

img_t = torch.from_numpy(img).permute(2, 0, 1)

C, H, W = img_t.shape
# beta = 0.03

# 主要图像
fre = torch.fft.fft2(img_t, dim=(1, 2))  # 变换得到的频域图数据是复数组成的
fre = torch.fft.fftshift(fre, dim=(1, 2))

showTensorImg(img_t[None])
showTensorImg(torch.abs(fre)[None] / fre.abs().mean())

# 获得空白背景的图像
blank = torch.ones_like(img_t)
blank_fre = torch.fft.fft2(blank, dim=(1, 2))
blank_fre = torch.fft.fftshift(blank_fre, dim=(1, 2))

showTensorImg(blank[None])
showTensorImg(torch.abs(blank_fre)[None])

# 替换低频信号
beta = 0.005
fre[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)] = \
    blank_fre[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)]

showTensorImg(torch.abs(fre)[None] / fre.abs().mean())

# 逆变换还原
ifre = torch.fft.ifftshift(fre, dim=[1, 2])  # 移动到原来位置
img_onlyphase = torch.abs(torch.fft.ifft2(ifre, dim=(1, 2)) ) # 还原为空间域图像
showTensorImg(img_onlyphase[None])

# 替换低频信号
beta = 0.01
fre[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)] = \
    blank_fre[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)]

showTensorImg(torch.abs(fre)[None] / fre.abs().mean())

# 逆变换还原
ifre = torch.fft.ifftshift(fre, dim=[1, 2])  # 移动到原来位置
img_onlyphase = torch.abs(torch.fft.ifft2(ifre, dim=(1, 2)) ) # 还原为空间域图像
showTensorImg(img_onlyphase[None])

# 替换低频信号
beta = 0.05
fre[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)] = \
    blank_fre[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)]

showTensorImg(torch.abs(fre)[None] / fre.abs().mean())

# 逆变换还原
ifre = torch.fft.ifftshift(fre, dim=[1, 2])  # 移动到原来位置
img_onlyphase = torch.abs(torch.fft.ifft2(ifre, dim=(1, 2)) ) # 还原为空间域图像
showTensorImg(img_onlyphase[None])

# 替换低频信号
beta = 0.1
fre[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)] = \
    blank_fre[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)]

showTensorImg(torch.abs(fre)[None] / fre.abs().mean())

# 逆变换还原
ifre = torch.fft.ifftshift(fre, dim=[1, 2])  # 移动到原来位置
img_onlyphase = torch.abs(torch.fft.ifft2(ifre, dim=(1, 2)) ) # 还原为空间域图像
showTensorImg(img_onlyphase[None])