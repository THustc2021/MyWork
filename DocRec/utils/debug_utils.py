import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

def show_without_blank(mat, cmap=None):
    def _show(mat):
        fig, ax = plt.subplots()
        ax.imshow(mat, aspect='equal', cmap=cmap)
        ax.axis('off')
        # 去除图像周围的白边
        height, width = mat.shape[:2]
        # 如果dpi=300，那么图像大小=height*width
        fig.set_size_inches(width / 100.0, height / 100.0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.show()
    if len(mat.shape) == 3 and mat.shape[-1] == 2:
        _show(mat[..., 0])
        _show(mat[..., 1])
    else:
        _show(mat)

def showTensorImg(inp, idx=0, cmap=None):

    inp = inp.detach().cpu().numpy()[idx]
    inp = inp.transpose(1, 2, 0)

    if inp.max() <= 1:
        inp = (inp*255.).astype(np.uint8)

    show_without_blank(inp, cmap)

def recoverImg(inp, bm, idx=0):

    inp = inp.detach().cpu()
    bm = bm.detach().cpu().permute(0, 2, 3, 1)

    res = F.grid_sample(inp, bm, align_corners=False)
    showTensorImg(res, idx=idx)