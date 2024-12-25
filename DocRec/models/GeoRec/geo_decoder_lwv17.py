import torch
import torch.nn.functional as F
from torch import nn

from models.Extractor.encoder import FeatureEncoder
from utils.common import reload_model
from models.GeoRec.basic.CAM import *

class TransCAM(nn.Module):

    def __init__(self, hq_dim, lq_dim, inner_dim, upsample=16):
        super(TransCAM, self).__init__()

        self.conv_hq = nn.Sequential(
            nn.Conv2d(hq_dim, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
            nn.Conv2d(inner_dim, inner_dim, 1, 1, 0)
        )
        self.conv_lq = nn.Conv2d(lq_dim, lq_dim, 3, 1, 1)
        self.fc_lq = nn.Linear(upsample*upsample, 1)
        self.conv_lq_k = nn.Sequential(
            nn.Conv2d(lq_dim, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
            nn.Conv2d(inner_dim, inner_dim, 1, 1, 0)
        )
        self.conv_lq_v = nn.Sequential(
            nn.Conv2d(lq_dim, inner_dim, 3, 1, 1),
            nn.BatchNorm2d(inner_dim),
            nn.ReLU(),
            nn.Conv2d(inner_dim, inner_dim, 1, 1, 0)
        )

        self.fc = nn.Linear(inner_dim, upsample*upsample*9)

    def forward(self, hqf, lqf):

        b, c, h, w = hqf.shape
        _, bc, bh, bw = lqf.shape
        #
        nh, nw = bh // h, bw // w
        lqf = self.conv_lq(lqf)
        lqf = lqf.reshape(b, bc, h, nh, w, nw).transpose(3, 4).flatten(1, 3) # 划分窗口 b, bc * h * w, nh, nw
        lqf = self.fc_lq(lqf.flatten(-2, -1)).view(b, bc, h, w)
        # 使用深层特征的每个像素位置，去匹配高频窗口，匹配上的窗口具有较大相似度
        # 获得每个小窗口的特征键表示
        lqfk = self.conv_lq_k(lqf)
        # 匹配开始
        hqfq = self.conv_hq(hqf)
        hqfq = hqfq.flatten(-2, -1).transpose(1, 2) # b, h*w, cin
        lqfk = lqfk.flatten(-2, -1) # b, cin, h*w
        sim = F.softmax(torch.bmm(hqfq, lqfk), dim=1) # b, h*w, h*w，找到最匹配的局部窗口
        # 获取每个小窗口的特征值表示
        lqfv = self.conv_lq_v(lqf).flatten(-2, -1).transpose(1, 2)  # b, h*w, cin
        attn = F.tanh(torch.bmm(sim, lqfv))    # b, h*w, cin
        # 获取上采样倍数
        upsample_ratio = self.fc(attn)  # b, h*w, s*s
        # 折叠
        hqf_fold = F.unfold(hqf, [3, 3], padding=1)
        hqf_fold = hqf_fold.view(b, c, 9, 1, 1, h, w)
        upsample_ratio = upsample_ratio.view(b, 1, h, w, 9, nh, nw)
        upsample_ratio = upsample_ratio.permute(0, 1, 4, 5, 6, 2, 3)

        up_flow = torch.sum(upsample_ratio * hqf_fold, dim=2)  # (N, 2, 16, 16, H, W)，使用mask在flow的3*3邻域内进行加权结合
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(b, c, bh, bw)

class GeoDecoder(nn.Module):
    def __init__(self,  norm_fn, acf):
        super(GeoDecoder, self).__init__()
        if norm_fn == "batch":
            norm = nn.BatchNorm2d
        elif norm_fn == "instance":
            norm = nn.BatchNorm2d
        else:
            raise Exception("wrong norm type!")

        # 合并高低频特征
        self.attn = AttentionBlock(256, 32, 32, 32, norm, acf, type=0)
        self.upsample = TransCAM(32, 32, 32, upsample=16)
        self.post_p = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            norm(32),
            acf,
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0),
            nn.Hardtanh(-1, 1)
        )

    def forward(self, x0, x1, x2, x3, f, x):

        # 坐标映射
        b, c, h, w = f.shape

        coords = gen_coords(h, w)[None].to(f)
        pn_embed = get_positional_embeddings(coords[:, 0], coords[:, 1], 32 // 2).repeat(b, 1, 1, 1)
        coords_feature = self.attn(pn_embed, f)

        grid_feature = self.upsample(coords_feature, x0)
        grid = self.post_p(grid_feature)

        return grid

class GeoRec(nn.Module):
    def __init__(self, in_channel=3, norm_fn="batch", acf=nn.LeakyReLU()):
        super(GeoRec, self).__init__()
        self.norm_fn = norm_fn
        self.acf = acf
        self.extractor = FeatureEncoder(in_channel, norm_fn=norm_fn, acf=acf)
        self.geo_decoder = GeoDecoder(norm_fn, self.acf)

    def load_model(self, state_dict, map_location="cuda", not_use_parrel_trained=True):
        reload_model(self, state_dict, map_device=map_location, not_use_parrel_trained=not_use_parrel_trained)

    def forward(self, x):
        x0, x1, x2, x3, f = self.extractor(x)
        pred = self.geo_decoder(x0, x1, x2, x3, f, x)
        return pred


if __name__ == '__main__':
    model = GeoRec()
    from torchinfo import summary

    summary(model, (2, 3, 224, 224), depth=1, device="cpu")