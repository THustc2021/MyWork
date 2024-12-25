import os
import cv2

if __name__ == '__main__':

    img_path = "/home/xtanghao/THPycharm/dataset/DIR300/dist"
    msk_path = "/home/xtanghao/THPycharm/dataset/DIR300/color_mask"

    save_path = "/home/xtanghao/THPycharm/dataset/DIR300/seg"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for d in os.listdir(img_path):
        img = cv2.imread(os.path.join(img_path, d))
        msk = cv2.imread(os.path.join(msk_path, d[:-4]+"_color_mask.png"), cv2.IMREAD_GRAYSCALE)
        msk = (msk>128).astype(int)

        res = img * msk[:,:, None]

        cv2.imwrite(os.path.join(save_path, d), res)