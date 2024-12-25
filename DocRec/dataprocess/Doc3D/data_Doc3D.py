import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import random
import cv2, torch
import hdf5storage as h5
import numpy as np
from torch.utils.data import Dataset


class DocIll_dataset(Dataset):

    def __init__(self, root_dir, mode, img_path_name, label_path_name,
                 msk_label_name=None, img_transform=None, cvt_hsv=False,
                 mask_input=False, mask_output=False):
        data = []
        assert mode in ['train', 'valid'], " mode must in ['train','valid']"
        with open(os.path.join(root_dir, "ill_" + mode + '.txt'), "r", encoding="utf-8") as f:
            paths = [p.split("\n")[0] for p in f.readlines()]
        for ps in paths:
            path0, path1 = ps.split(",")
            if not mask_input and not mask_output:
                if not path0.endswith(".png"):
                    path0 = path0 + ".png"
                if not path1.endswith(".png"):
                    path1 = path1 + ".png"
                data.append((os.path.join(root_dir, img_path_name, path0 ),
                             os.path.join(root_dir, label_path_name, path1)))
            else:
                if not path0.endswith(".png"):
                    path0 = path0 + ".png"
                if not path1.endswith(".png"):
                    path1 = path1 + ".png"
                data.append((os.path.join(root_dir, img_path_name, path0),
                             os.path.join(root_dir, label_path_name, path1),
                             os.path.join(root_dir, msk_label_name, path0 if msk_label_name=="msk_rec" else path1)))
        random.shuffle(data)
        self.mode = mode
        self.data = data
        self.img_transform = img_transform
        self.cvt_hsv = cvt_hsv

        self.mask_input = mask_input
        self.mask_output = mask_output

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.data)

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        distorted_image_path = self.data[index][0]
        corrected_image_path = self.data[index][1]

        if self.cvt_hsv:
            distorted_image = cv2.cvtColor(cv2.imread(distorted_image_path), cv2.COLOR_BGR2HSV)
            corrected_image = cv2.cvtColor(cv2.imread(corrected_image_path), cv2.COLOR_BGR2HSV)
        else:
            distorted_image = cv2.cvtColor(cv2.imread(distorted_image_path), cv2.COLOR_BGR2RGB)
            corrected_image = cv2.cvtColor(cv2.imread(corrected_image_path), cv2.COLOR_BGR2RGB)

        if self.img_transform != None:
            if not self.mask_input and not self.mask_output:
                res = self.img_transform(image=distorted_image, mask=corrected_image)
                distorted_image = res["image"]
                corrected_image = res["mask"]
            else:
                msk_image = cv2.cvtColor(cv2.imread(self.data[index][2]), cv2.COLOR_BGR2GRAY)
                res = self.img_transform(image=distorted_image, masks=[corrected_image, msk_image])
                distorted_image = res["image"]
                corrected_image, mask = res["masks"]
                if self.mask_input:
                    distorted_image = distorted_image - distorted_image * (mask[None] / 255)

        if not self.mask_output:
            if random.random() > 0.05:
                return distorted_image / 255., corrected_image / 255.
            else:
                return corrected_image / 255., corrected_image / 255.
        else:
            if random.random() > 0.05:
                return distorted_image / 255., corrected_image / 255., mask / 255
            else:
                return corrected_image / 255., corrected_image / 255., mask / 255

class FSDSRD_dataset(Dataset):

    def __init__(self, root_dir, mode, img_transform=None, need_shadow=False):
        data = []
        assert mode in ['train', 'valid'], " mode must in ['train','valid']"
        with open(os.path.join(root_dir, "ill_" + mode + '.txt'), "r", encoding="utf-8") as f:
            paths = [p.split("\n")[0] for p in f.readlines()]
        for ps in paths:
            path0, path1 = ps.split(",")
            if not path0.endswith(".png"):
                path0 = path0 + ".png"
            if not path1.endswith(".png"):
                path1 = path1 + ".png"
            data.append((os.path.join(root_dir, "img", path0),
                         os.path.join(root_dir, "gt", path1),
                         os.path.join(root_dir, "shadow", path0)))

        random.shuffle(data)
        self.mode = mode
        self.data = data
        self.img_transform = img_transform
        self.need_shadow = need_shadow

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.data)

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        distorted_image_path = self.data[index][0]
        corrected_image_path = self.data[index][1]
        shadow_image_path = self.data[index][2]

        distorted_image = cv2.cvtColor(cv2.imread(distorted_image_path), cv2.COLOR_BGR2RGB)
        corrected_image = cv2.cvtColor(cv2.imread(corrected_image_path), cv2.COLOR_BGR2RGB)
        shadow_image = cv2.cvtColor(cv2.imread(shadow_image_path), cv2.COLOR_BGR2RGB)

        if self.img_transform != None:
            res = self.img_transform(image=distorted_image, masks=[corrected_image, shadow_image])
            distorted_image = res["image"]
            corrected_image, shadow_image = res["masks"]

        if self.need_shadow:
            return distorted_image / 255., corrected_image / 255., shadow_image / 255.
        return distorted_image / 255., corrected_image / 255.

class Doc3d_Geo_dataset(Dataset):
    """ 此dataset的label是反向映射图，原始输入是背景去除后的拍照文档图像

    """

    def __init__(self, root_dir, mode, img_path_name="rembg", img_transform=None):
        super(Doc3d_Geo_dataset, self).__init__()

        # 根据data_txt找到所有文件路径
        # 这里直接读入背景去除后的图像
        data = []
        with open(os.path.join(root_dir, "geo_" + mode + '.txt'), "r", encoding="utf-8") as f:
            paths = [p.split("\n")[0] for p in f.readlines()]
        for path in paths:
            data.append((os.path.join(root_dir, img_path_name, path + ".png"),
                         os.path.join(root_dir, "bm", path + ".mat")))
        random.shuffle(data)
        # 汇总
        self.data = data
        self.img_transform = img_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # 读取图像、mask和label，归一化图像（重要），标签不用归一化
        img = cv2.cvtColor(cv2.imread(self.data[index][0]), cv2.COLOR_BGR2RGB)
        try:
            label = h5.loadmat(self.data[index][1])["bm"].astype(float)
        except:
            print(self.data[index][1])
            raise Exception()
        # 归一化label到 [-1,1]
        label = label.astype(float)
        label = label / np.array([label.shape[0], label.shape[1]])
        label = (label - 0.5) * 2

        if self.img_transform != None:
            res = self.img_transform(image=img, mask=label)
            img, label = res["image"], res["mask"]

        return img / 255.0, label.float()

class Doc3D_All_Dataset(Dataset):

    def __init__(self, root_dir, mode, img_path_name, img_transform=None, use_msk=False):
        data = []
        with open(os.path.join(root_dir, "ill_" + mode + '.txt'), "r", encoding="utf-8") as f:
            paths = [p.split("\n")[0] for p in f.readlines()]
        for ps in paths:
            path0, path1 = ps.split(",")
            data.append((os.path.join(root_dir, img_path_name, path0 + ".png"),
                         os.path.join(root_dir, "geo_gt", path1 + ".png"),
                         os.path.join(root_dir, "bm", path1 + ".mat"),
                         os.path.join(root_dir, "wc", path1 + ".exr"),
                         os.path.join(root_dir, "msk", path1 + ".png"),
                         os.path.join(root_dir, "ill_gt", path1 + ".png")
                         ))
        random.shuffle(data)
        self.mode = mode
        self.data = data
        self.img_transform = img_transform
        self.use_msk = use_msk

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.data)

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        distorted_image_path = self.data[index][0]
        corrected_image_path = self.data[index][1]
        msk_image_path = self.data[index][4]
        ill_image_path = self.data[index][5]

        distorted_image = cv2.cvtColor(cv2.imread(distorted_image_path), cv2.COLOR_BGR2RGB)
        corrected_image = cv2.cvtColor(cv2.imread(corrected_image_path), cv2.COLOR_BGR2RGB)
        if self.use_msk:
            msk_image = cv2.imread(msk_image_path, cv2.IMREAD_GRAYSCALE)
        ill_image = cv2.cvtColor(cv2.imread(ill_image_path), cv2.COLOR_BGR2RGB)

        #获得bm
        label = h5.loadmat(self.data[index][2])["bm"].astype(float)
        # 归一化label到 [-1,1]
        label = label / np.array([label.shape[0], label.shape[1]])
        label = (label - 0.5) * 2

        if self.img_transform != None:
            if self.use_msk:
                res = self.img_transform(image=distorted_image, masks=[corrected_image, label, ill_image, msk_image])
                distorted_image = res["image"]
                corrected_image, label, ill_image, msk_image = res["masks"]
                distorted_image = torch.cat([distorted_image, msk_image[None]], dim=0)
            else:
                res = self.img_transform(image=distorted_image, masks=[corrected_image, label, ill_image])
                distorted_image = res["image"]
                corrected_image, label, ill_image = res["masks"]

        return distorted_image / 255., corrected_image / 255., label.float(), ill_image / 255.

class WarpDoc_Dataset(Dataset):

    def __init__(self, root_dir, mode, img_path_name, label_path_name, img_transform=None, exclude=[]):
        data = []
        assert mode in ['train', 'valid'], " mode must in ['train','valid']"
        with open(os.path.join(root_dir, "ill_" + mode + '.txt'), "r", encoding="utf-8") as f:
            paths = [p.split("\n")[0] for p in f.readlines()]
        for ps in paths:
            # 排除路径
            flag = 0
            for e in exclude:
                if ps.startswith(e):
                    flag = 1
                    break
            if flag:
                continue
            # 记录数据
            path0, path1 = ps.split(",")
            path0 = path0.split("/")
            path0.insert(1, "seg")
            path0 = os.path.join(*path0)
            data.append((os.path.join(root_dir, img_path_name, path0),
                         os.path.join(root_dir, label_path_name, path1),
                        ))
        random.shuffle(data)
        self.mode = mode
        self.data = data
        self.img_transform = img_transform

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.data)

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        distorted_image_path = self.data[index][0]
        corrected_image_path = self.data[index][1]

        distorted_image = cv2.cvtColor(cv2.imread(distorted_image_path), cv2.COLOR_BGR2RGB)
        corrected_image = cv2.cvtColor(cv2.imread(corrected_image_path), cv2.COLOR_BGR2RGB)

        if self.img_transform != None:
            res = self.img_transform(image=distorted_image, mask=corrected_image)
            distorted_image = res["image"]
            corrected_image = res["mask"]

        return distorted_image / 255., corrected_image / 255.

class DocDea_Dataset(Dataset):

    def __init__(self, root_dir, mode, img_transform=None, return_uv=True):
        data = []
        assert mode in ['train', 'valid'], " mode must in ['train','valid']"
        with open(os.path.join(root_dir, mode + '.txt'), "r", encoding="utf-8") as f:
            paths = [p.split("\n")[0] for p in f.readlines()]
        for ps in paths:
            idx, path0, path1, coords = ps.split(",")
            data.append((os.path.join(root_dir, "img", str(idx) + ".png"),
                         path0, path1, coords.split(" ")))
        random.shuffle(data)
        self.mode = mode
        self.data = data
        self.img_transform = img_transform
        self.return_uv = return_uv

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.data)

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        distorted_image_path = self.data[index][0]
        corrected_image_path = self.data[index][1]
        bm_path = self.data[index][2]
        ri, ci, h, w = self.data[index][3]

        distorted_image = cv2.cvtColor(cv2.imread(distorted_image_path), cv2.COLOR_BGR2RGB)
        corrected_image = cv2.imread(corrected_image_path)  # label存的时候没有改变颜色

        # 获得bm
        label = h5.loadmat(bm_path)["bm"].astype(float)
        # 归一化label到 [-1,1]
        label = label / np.array([label.shape[0], label.shape[1]])
        label = (label - 0.5) * 2

        if self.return_uv:
            uv_path = os.path.join(os.path.dirname(bm_path).replace("bm", "uv"), os.path.basename(bm_path).replace("mat", "exr"))
            uv = cv2.imread(uv_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            uv = 2 * np.stack([uv[:, :, 2], 1 - uv[:, :, 1]], axis=2) - 1
        else:
            uv = np.ones_like(label)

        # 预裁剪
        corrected_image = corrected_image[int(ri):(int(ri)+int(h)), int(ci):(int(ci)+int(w))]

        if self.img_transform != None:
            res = self.img_transform(image=distorted_image, masks=[corrected_image, label, uv])
            distorted_image = res["image"]
            corrected_image, label, uv = res["masks"]

        return distorted_image / 255., corrected_image / 255., label.float(), uv

import albumentations.augmentations.functional as AF
class randomshadowDataset(Dataset):
    """
        让鉴别器学会鉴别图像区域是否有阴影。
        有阴影的图像和无阴影的图像会同时输入给鉴别器，鉴别器通过比较二者的表示来判断有阴影的区域。
        为了让模型不是简单地对二者的表示做减法，我们会在粘贴完阴影后添加一些非阴影噪声
        0表示无阴影，1表示有阴影
    """

    def __init__(self, unpaired_img_dirs, paired_img_dirs, mode, pre_transform, tile=(7, 7), img_transform=None):

        # train和val的txt文件要记录所有路径信息
        data = []
        for img_dir, img_path_name in unpaired_img_dirs:
            with open(os.path.join(img_dir, "ill_" + mode + ".txt"), "r", encoding="utf-8") as f:
                paths = [os.path.join(img_dir, img_path_name, p.split(",")[1].strip()) for p in f.readlines()]
            data.extend(paths)
        for img_dir, img_path_name, label_path_name in paired_img_dirs:
            with open(os.path.join(img_dir, "ill_" + mode + '.txt'), "r", encoding="utf-8") as f:
                paths = [p.split("\n")[0] for p in f.readlines()]
            for ps in paths:
                path0, path1 = ps.split(",")
                data.append((os.path.join(img_dir, img_path_name, path0 + ".png"),
                             os.path.join(img_dir, label_path_name, path1 + ".png")))
        random.shuffle(data)
        random.shuffle(data)

        self.data = data
        self.tile = tile

        self.pre_transform = pre_transform
        self.img_transform = img_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = self.data[index]
        if len(item) == 2:  # paired_data
            # 读取图片
            img = cv2.cvtColor(cv2.imread(item[0]), cv2.COLOR_BGR2RGB)
            alb = cv2.cvtColor(cv2.imread(item[1]), cv2.COLOR_BGR2RGB)
            res = self.pre_transform(image=img, mask=alb)
            img, alb = res["image"], res["mask"]
            if random.random() >= 0.5:
                label = np.ones(self.tile)
            # 随机替换
            else:
                # stack mask
                i_mask = np.ones((img.shape[0], img.shape[1], 1), dtype=np.uint8)
                img_with_msk = np.concatenate([img, i_mask], axis=2)
                a_mask = np.zeros((alb.shape[0], alb.shape[1], 1), dtype=np.uint8)
                alb_with_msk = np.concatenate([alb, a_mask], axis=2)
                # 粘贴
                paste_time = random.randint(1, 3)
                for _ in range(paste_time):
                    h = random.randint(10, img.shape[0] - 10)
                    w = random.randint(10, img.shape[1] - 10)
                    x = random.randint(5, img.shape[0] - h - 5)
                    y = random.randint(5, img.shape[1] - w - 5)
                    if (alb_with_msk[x:x+h, y:y+w] == img_with_msk[x:x+h, y:y+w]).any():    # 如内容本来一样就不复制
                        continue
                    alb_with_msk[x:x+h, y:y+w] = img_with_msk[x:x+h, y:y+w]
                # 检索与生成label
                img = alb_with_msk[:, :, :3]
                alb = alb
                # 划分窗口方式
                label = alb_with_msk[:, :, 3].reshape(self.tile[0], img.shape[0] // self.tile[0],
                                                      self.tile[1], img.shape[1] // self.tile[1]).transpose(0, 2, 1, 3).\
                                              reshape(self.tile[0], self.tile[1], -1).sum(axis=2)
                label[label>0] = 1
        else:
            # 随机位置阴影
            try:
                img = cv2.cvtColor(cv2.imread(item), cv2.COLOR_BGR2RGB)
            except:
                print(item)
                raise Exception()
            alb = img
            res = self.pre_transform(image=img, mask=alb)
            img, alb = res["image"], res["mask"]
            mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
            # 粘贴
            paste_time = random.randint(1, 3)
            for _ in range(paste_time):
                h = random.randint(10, img.shape[0] - 10)
                w = random.randint(10, img.shape[1] - 10)
                x = random.randint(5, img.shape[0] - h - 5)
                y = random.randint(5, img.shape[1] - w - 5)
                # 随机阴影多边形的顶点
                vn = random.randint(3, 5)   # 边数
                vs = list(zip(random.choices(list(range(w)), k=vn), random.choices(list(range(h)), k=vn)))
                # 随机阴影
                try:
                    img[x:x + h, y:y + w] = AF.add_shadow(img[x:x + h, y:y + w], [[np.array(vs)]])
                except Exception as e:
                    print(e)
                    print(f"wrong vs {vs}, for vn {vn}, h, w {h}, {w}")
                    continue
                mask[x:x + h, y:y + w] = 1
            # 检索与生成label
            label = mask.reshape(self.tile[0], img.shape[0] // self.tile[0],
                              self.tile[1], img.shape[1] // self.tile[1]).transpose(0, 2, 1, 3). \
                              reshape(self.tile[0], self.tile[1], -1).sum(axis=2)
            label[label > 0] = 1

        if self.img_transform != None:
            res = self.img_transform(image=img, mask=alb)
            img = res["image"]
            alb = res["mask"]

        return img / 255., alb / 255., label.astype(np.float32)

class D_dataset(Dataset):
    def __init__(self, root_dir, mode, img_path_name, label_path_name, img_transform=None, use_foreground=False):
        data = []
        assert mode in ['train', 'valid'], " mode must in ['train','valid']"
        with open(os.path.join(root_dir, "ill_" + mode + '.txt'), "r", encoding="utf-8") as f:
            paths = [p.split("\n")[0] for p in f.readlines()]
        for paths in paths:
            path0, path1 = paths.split(",")
            data.append((os.path.join(root_dir, img_path_name, path0 + ".png"),
                         os.path.join(root_dir, label_path_name, path1 + ".png")))

        self.mode = mode
        self.data = data
        self.img_transform = img_transform
        self.use_foreground = use_foreground

    def __len__(self):
        """Returns the total number of image files."""
        return len(self.data)

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        distorted_image_path = self.data[index][0]
        corrected_image_path = self.data[index][1]

        distorted_image = cv2.cvtColor(cv2.imread(distorted_image_path), cv2.COLOR_BGR2RGB)
        corrected_image = cv2.cvtColor(cv2.imread(corrected_image_path), cv2.COLOR_BGR2RGB)
        if self.use_foreground:
            foreground_mask = cv2.cvtColor(cv2.imread(distorted_image_path.replace("img", "foreground_mask")),
                                           cv2.COLOR_BGR2RGB)
            corrected_image = corrected_image * foreground_mask

        if self.img_transform != None:
            res = self.img_transform(image=distorted_image, mask=corrected_image)
            distorted_image = res["image"]
            corrected_image = res["mask"]

        if random.random() >= 0.5:
            return distorted_image / 255., 0.
        else:
            return corrected_image / 255., 1.