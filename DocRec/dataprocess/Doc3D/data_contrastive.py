import os
import random
from os.path import join as pjoin

import cv2
import torch

from torch.utils.data import Dataset


class ContrastiveDataset(Dataset):
    
    def __init__(self, root_dir, crop_transform, img_transform=None):
        super(ContrastiveDataset, self).__init__()

        data = []
        for d in root_dir:
            data.extend([(pjoin(d, "img", dd), pjoin(d, "msk", dd)) for dd in os.listdir(pjoin(d, "img"))])
        random.shuffle(data)
        self.data = data
        self.crop_transform = crop_transform
        self.img_transform = img_transform

    def get_fourier_transform(self, img_t, beta=0.008):
        _, H, W = img_t.shape

        # 主要图像
        fre = torch.fft.fft2(img_t, dim=(1, 2))  # 变换得到的频域图数据是复数组成的
        fre = torch.fft.fftshift(fre, dim=[1, 2])

        # 获得空白背景的图像
        blank = torch.ones_like(img_t)
        blank_fre = torch.fft.fft2(blank, dim=(1, 2))
        blank_fre = torch.fft.fftshift(blank_fre, dim=(1, 2))

        # 替换低频信号
        fre[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)] = \
            blank_fre[:, int(H/2 - beta * H/2):int(H/2 + beta * H/2), int(W/2-beta*W/2):int(W/2+beta*W/2)]
        # fre[:, :int(beta * H), :int(beta * W)] = blank_fre[:, :int(beta * H), :int(beta * W)]

        # 逆变换还原
        fre = torch.fft.ifftshift(fre, dim=[1, 2])  # 移动到原来位置
        img_onlyphase = torch.abs(torch.fft.ifft2(fre, dim=(1, 2)) ) # 还原为空间域图像
        return img_onlyphase

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        sample_path = self.data[index][0]
        sample_msk_path = self.data[index][1]

        sample = cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB)
        sample_msk = cv2.imread(sample_msk_path, cv2.IMREAD_UNCHANGED) // 255

        sample_res = self.crop_transform(image=sample, mask=sample_msk)
        sample = sample_res["image"]
        sample_msk = sample_res["mask"]

        t0_res = self.img_transform(image=sample, mask=sample_msk)
        t1_res = self.img_transform(image=sample, mask=sample_msk)

        sample_t0 = t0_res["image"] * t0_res["mask"] / 255.
        sample_t1 = t1_res["image"] / 255.

        sample_t1 = self.get_fourier_transform(sample_t1) * t1_res["mask"]

        return sample_t0, sample_t1


class ContrastiveDatasetv2(Dataset):

    def __init__(self, root_dir, crop_transform, img_transform=None):
        super(ContrastiveDatasetv2, self).__init__()

        data = []
        for d in root_dir:
            ddata = []
            for dd in os.listdir(pjoin(d, "img")):
                if os.path.isdir(pjoin(d, "img", dd)):
                    ddata.extend([(pjoin(d, "img", dd, ddd), pjoin(d, "msk", dd, ddd)) for ddd in os.listdir(pjoin(d, "img", dd))])
                else:
                    ddata.append((pjoin(d, "img", dd), pjoin(d, "msk", dd)))
            data.extend(ddata)
        random.shuffle(data)
        # 分成两摞
        self.data1 = data[:len(data)//2]    # data1是小于data2的，没问题
        self.data2 = data[len(data)//2:]
        #
        self.crop_transform = crop_transform
        self.img_transform = img_transform

    def get_fourier_transform(self, img_t, img_t1, beta=0.008):

        beta = random.random() * 0.092 + 0.008

        _, H, W = img_t.shape

        # 主要图像
        fre = torch.fft.fft2(img_t, dim=(1, 2))  # 变换得到的频域图数据是复数组成的
        fre = torch.fft.fftshift(fre, dim=[1, 2])

        # 获得空白背景的图像
        fre1 = torch.fft.fft2(img_t1, dim=(1, 2))
        fre1 = torch.fft.fftshift(fre1, dim=(1, 2))

        # clone
        fre0 = fre.clone()

        # 替换低频信号
        fre[:, int(H / 2 - beta * H / 2):int(H / 2 + beta * H / 2),
        int(W / 2 - beta * W / 2):int(W / 2 + beta * W / 2)] = \
            fre1[:, int(H / 2 - beta * H / 2):int(H / 2 + beta * H / 2),
            int(W / 2 - beta * W / 2):int(W / 2 + beta * W / 2)]
        fre1[:, int(H / 2 - beta * H / 2):int(H / 2 + beta * H / 2),
        int(W / 2 - beta * W / 2):int(W / 2 + beta * W / 2)] = \
            fre0[:, int(H / 2 - beta * H / 2):int(H / 2 + beta * H / 2),
            int(W / 2 - beta * W / 2):int(W / 2 + beta * W / 2)]

        # 逆变换还原
        fre = torch.fft.ifftshift(fre, dim=[1, 2])  # 移动到原来位置
        img_onlyphase = torch.abs(torch.fft.ifft2(fre, dim=(1, 2)))  # 还原为空间域图像
        fre1 = torch.fft.ifftshift(fre1, dim=[1, 2])  # 移动到原来位置
        img_onlyphase1 = torch.abs(torch.fft.ifft2(fre1, dim=(1, 2)))  # 还原为空间域图像

        return img_onlyphase, img_onlyphase1    # 用1低频改造0，用0低频改造1

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        sample_path = self.data1[index][0]
        sample_msk_path = self.data1[index][1]

        sample_path1 = self.data2[index][0]
        sample_msk_path1 = self.data2[index][1]

        sample = cv2.cvtColor(cv2.imread(sample_path), cv2.COLOR_BGR2RGB)
        sample_msk = cv2.imread(sample_msk_path, cv2.IMREAD_UNCHANGED) // 255
        sample1 = cv2.cvtColor(cv2.imread(sample_path1), cv2.COLOR_BGR2RGB)
        sample_msk1 = cv2.imread(sample_msk_path1, cv2.IMREAD_UNCHANGED) // 255

        sample_res = self.crop_transform(image=sample, mask=sample_msk)
        sample = sample_res["image"]
        sample_msk = sample_res["mask"]

        sample_res1 = self.crop_transform(image=sample1, mask=sample_msk1)
        sample1 = sample_res1["image"]
        sample_msk1 = sample_res1["mask"]

        t0_res = self.img_transform(image=sample, mask=sample_msk)
        t1_res = self.img_transform(image=sample1, mask=sample_msk1)

        sample_t0 = t0_res["image"] * t0_res["mask"] / 255.
        sample_t1 = t1_res["image"] * t1_res["mask"] / 255.

        sample_t01, sample_t10 = self.get_fourier_transform(sample_t0, sample_t1)
        # sample_t01 = sample_t01 * t0_res["mask"]
        # sample_t11 = sample_t11 * t1_res["mask"]

        return sample_t0, sample_t1, sample_t10, sample_t01

if __name__ == '__main__':

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    ContrastiveDatasetv2(["/home/xtanghao/THPycharm/dataset/diw5k"], crop_transform=A.CropNonEmptyMaskIfExists(height=448, width=488),
                       img_transform=
                       A.Compose([
                           A.GaussianBlur(),
                           ToTensorV2(transpose_mask=True)
                       ]))[0]