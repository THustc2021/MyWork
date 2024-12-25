import itertools
import os

import cv2
import torch
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import Dataset

from os.path import join as pjoin

class JigsawDataset(Dataset):

    def __init__(self, root_dirs, patch_shape, img_shape, crop_transform, img_transform=None):
        super(JigsawDataset, self).__init__()

        data = []
        for d in root_dirs:
            for dd in os.listdir(d):
                if os.path.isdir(pjoin(d, dd)):
                    data.extend([pjoin(d ,dd, ddd) for ddd in os.listdir(pjoin(d, dd))])
                else:
                    data.append(pjoin(d, dd))
        self.data = data
        self.crop_transform = crop_transform
        self.img_t = img_transform

        x_num, y_num = img_shape[1] // patch_shape[1], img_shape[0] // patch_shape[0]
        self.im_to_patches = torch.nn.Unfold(patch_shape, stride=patch_shape)
        self.patches_to_im = torch.nn.Fold(
            output_size=img_shape,
            kernel_size=patch_shape,
            stride=patch_shape
        )
        self.x_num = x_num
        self.y_num = y_num
        self.idxs = np.linspace(0, x_num * y_num - 1, x_num * y_num)
        self.img_shape = img_shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        img_p = self.data[index]
        img_ori = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)

        img_ori = self.crop_transform(image=img_ori)["image"] / 255.

        # 获取img的部分
        H, W = img_ori.shape[1:]
        h, w = self.img_shape
        img = img_ori[:, (H-h)//2:-(H-h)//2, (W-w)//2:-(W-w)//2]

        patches = self.im_to_patches(img[None])

        # 记录合法索引，为了便捷和提高学习效果，我们将重复的patch遮挡掉
        # p = rearrange(img, 'c (h p1) (w p2) -> (h w) (p1 p2) c', h=self.y_num, w=self.x_num)
        # p = p.sum(dim=2).var(dim=1)
        # not_empty_idx = (p != 0).flatten()
        distances = F.pdist(patches[0].transpose(0, 1), p=2)  # 使用欧氏距离
        # 根据距离构建对称矩阵
        num_samples = patches.size(2)
        # 构建全零矩阵
        symmetric_matrix = torch.zeros(num_samples, num_samples)
        # 根据距离填充对称矩阵
        row, col = torch.triu_indices(num_samples, num_samples, offset=1)
        symmetric_matrix[row, col] = distances
        # 使用对称性填充下三角部分
        symmetric_matrix = symmetric_matrix + symmetric_matrix.t()
        not_repeated_idx = ((symmetric_matrix == 0).sum(dim=1) == 1)

        # 产生打乱结果，没有打乱的部分则为mask的部分
        shuffle_ratio = random.random() * 0.6 + 0.2
        shuffle_idx = np.random.choice(self.idxs, size=int(self.idxs.shape[0] * shuffle_ratio),
                                       replace=False)  # 随机选择一定数量的patch
        random_indices = np.random.choice(shuffle_idx, size=shuffle_idx.shape[0], replace=False)  # 对随机选择出来的patch进行随机打

        #
        labels = torch.from_numpy(self.idxs.copy())
        labels[shuffle_idx] = labels[random_indices]
        # msk_indices = np.random.choice(self.idxs, size=int(self.x_num * self.y_num * self.mask_ratio), replace=False)
        random_indices = torch.from_numpy(random_indices).long()
        # msk_indices = torch.from_numpy(msk_indices).long().sort()[0]
        patches[:, :, shuffle_idx] = patches[:, :, random_indices]
        not_repeated_idx[shuffle_idx] = not_repeated_idx[random_indices]
        msk_indices = torch.ones(labels.shape[0], dtype=torch.bool)
        msk_indices[shuffle_idx] = False

        img_n = self.patches_to_im(patches)[0]

        # 贴回去
        img_change = img_ori.clone()
        img_change[:, (H-h)//2:-(H-h)//2, (W-w)//2:-(W-w)//2] = img_n

        if self.img_t != None:
            img_change = self.img_t(img_change)

        return img_change, labels.long(), not_repeated_idx, img_ori, msk_indices

class JigsawDatasetv2(Dataset):

    def __init__(self, root_dirs, patch_shape, img_shape, crop_transform, img_transform=None,
                 permute_path="/home/xtanghao/DocRec/bin/permutations/permutations_hamming_max_100.npy"):
        super(JigsawDatasetv2, self).__init__()

        data = []
        for d in root_dirs:
            for dd in os.listdir(d):
                if os.path.isdir(pjoin(d, dd)):
                    data.extend([pjoin(d ,dd, ddd) for ddd in os.listdir(pjoin(d, dd))])
                else:
                    data.append(pjoin(d, dd))
        self.data = data
        self.crop_transform = crop_transform
        self.img_t = img_transform

        x_num, y_num = img_shape[1] // patch_shape[1], img_shape[0] // patch_shape[0]
        self.im_to_patches = torch.nn.Unfold(patch_shape, stride=patch_shape)
        self.patches_to_im = torch.nn.Fold(
            output_size=img_shape,
            kernel_size=patch_shape,
            stride=patch_shape
        )
        self.x_num = x_num
        self.y_num = y_num
        self.idxs = np.linspace(0, x_num * y_num - 1, x_num * y_num)
        self.img_shape = img_shape

        self.plist = self._load_pertumation(permute_path)

    def __len__(self):
        return len(self.data)

    def _load_pertumation(self, pp):
        # all_perm = np.load(pp)
        # # from range [1,9] to [0,8]
        # if all_perm.min() == 1:
        #     all_perm = all_perm - 1
        #
        # return all_perm
        return np.array(list(itertools.permutations(list(range(4)), 4)))

    def __getitem__(self, index):

        img_p = self.data[index]
        img_ori = cv2.cvtColor(cv2.imread(img_p), cv2.COLOR_BGR2RGB)

        img_ori = self.crop_transform(image=img_ori)["image"] / 255.

        # 获取img的部分
        H, W = img_ori.shape[1:]
        h, w = self.img_shape
        img = img_ori[:, (H-h)//2:-(H-h)//2, (W-w)//2:-(W-w)//2]

        patches = self.im_to_patches(img[None])

        perm = random.randint(0, len(self.plist)-1)
        patches[:, :, self.idxs] = patches[:, :, self.plist[perm].tolist()]

        img_n = self.patches_to_im(patches)[0]

        # 贴回去
        img_change = img_ori.clone()
        img_change[:, (H-h)//2:-(H-h)//2, (W-w)//2:-(W-w)//2] = img_n

        if self.img_t != None:
            img_change = self.img_t(img_change)

        return img_change, perm, img_ori