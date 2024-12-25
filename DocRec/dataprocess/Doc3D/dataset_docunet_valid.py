import os.path

import cv2
from torch.utils.data import Dataset


class Valid_Dataset(Dataset):
    
    def __init__(self, root_dir="/home/xtanghao/THPycharm/dataset/DocUnet Benchmark", img_dir_name="seg/final",
                 img_transform=None):
        super(Valid_Dataset, self).__init__()

        targs_dir = os.path.join(root_dir, "scan")
        imgs_dir = os.path.join(root_dir, img_dir_name)

        img_names = os.listdir(imgs_dir)
        targ_names = os.listdir(targs_dir)

        data = []
        for targ_name in targ_names:
            name = targ_name.split(".png")[0]   # 前缀
            for img_name in img_names.copy():
                if img_name.split("_")[0] == name:
                    data.append((os.path.join(imgs_dir, img_name), os.path.join(targs_dir, targ_name)))
                    img_names.remove(img_name)
        print(f"test nums: {len(data)}")
        self.data = data
        self.img_transform = img_transform

    def __len__(self):
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