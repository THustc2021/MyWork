import random
import torch
import numpy as np
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.thinplatespline.batch import TPS


def set_determistic(seed=43):

    random.seed(seed)   # python random generator
    np.random.seed(seed)    # numpy random generator

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True   # 可能导致性能降低
    torch.backends.cudnn.benchmark = False  # 确保算法选择本身可复现。可能导致性能降低，并且算法本身可能也是不可复现的

def reload_model(model, path, map_device="cuda", not_use_parrel_trained=True, freeze=False):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(path, map_location=map_device)
    print(len(pretrained_dict.keys()))
    if not_use_parrel_trained:    # 不使用并行训练的模型
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == dict(model_dict.items())[k].shape}
    else:
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict and v.shape == dict(model_dict.items())[k[7:]].shape}
    print(len(pretrained_dict.keys()))
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    if freeze:  # 冻结模型参数
        for k, v in model.named_parameters():
            if k in pretrained_dict:
                v.requires_grad_(False)

    return model

def saveOutResult(inp, label, output, name, writer, output_is_bm=False, label_is_bm=False):
    pred = output
    if output_is_bm:
        pred = F.grid_sample(inp, pred.permute(0, 2, 3, 1), align_corners=True)
    if label_is_bm:
        label = F.grid_sample(inp, label.permute(0, 2, 3, 1), align_corners=True)
    inp = inp.detach().cpu().numpy().transpose(0, 2, 3, 1)
    pred = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
    label = label.detach().cpu().numpy().transpose(0, 2, 3, 1)
    fig, ax = plt.subplots(2, 3, dpi=100)
    for i in range(2):
        ax[i, 0].axis(False)
        ax[i, 0].title.set_text("img, size: {}".format(inp[i].shape))
        ax[i, 0].imshow(inp[i])
        ax[i, 1].axis(False)
        ax[i, 1].title.set_text("label, size: {}".format(label[i].shape))
        ax[i, 1].imshow(label[i])
        ax[i, 2].axis(False)
        ax[i, 2].title.set_text("output, size: {}".format(pred[i].shape))
        ax[i, 2].imshow(pred[i])
    writer.add_figure(f"Fig/{name}", fig)
    writer.flush()
    plt.close()

def warptps(img_t, X, Y):
    # X, Y 为normed tensor。所有张量均为四维
    tpsb = TPS(size=img_t.shape[2:], device=img_t.device)
    warped_grid_b = tpsb(X.to(img_t.device), Y.to(img_t.device)).to(img_t.device)  # tpsb(target, source)

    # 变形处理
    ten_wrp_b = torch.grid_sampler_2d(img_t, warped_grid_b, 0, 0, False)

    return ten_wrp_b


def gen_ori_points(points_num_per_axis, img_size, points_format=0):
    # points_format: 0->(h*w, 2), 1->(2, h, w)
    # 找到若干个控制点
    ys = np.linspace(0, img_size[0] - 1, points_num_per_axis[0], dtype=int)
    ys = ys.repeat(points_num_per_axis[0])
    xs = np.linspace(0, img_size[1] - 1, points_num_per_axis[1], dtype=int)
    xs = xs[:, None].repeat(points_num_per_axis[1], axis=1).transpose().flatten()
    points_ori = (list(xs), list(ys))  # ([x1, ...], [y1, ...])
    points_idx = points_ori[::-1]
    points_ori = list(zip(*points_ori))
    points_ori = 2 * np.array(points_ori, dtype=np.float32) / np.array([img_size[1] - 1, img_size[0] - 1]) - 1
    points_ori = torch.from_numpy(points_ori)
    if points_format == 1:
        points_ori = points_ori.view(points_num_per_axis, 2).permute(2, 0, 1)
    return points_idx, points_ori   # 返回原点索引（二元tuple，可用于tensor索引），以及原点tensor （2, h, w)