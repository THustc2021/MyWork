import math

import torch
import torch.nn.functional as F

from models.Extractor.encoder import FeatureEncoder, ResBlock
from models.GeoRec.basic.CAM import *
from utils.common import reload_model
from torch import nn

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttn(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(SpatialAttn, self).__init__()

        self.attn_conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, in_channels // reduction_ratio, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels // reduction_ratio, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, 1, 1, 1, 0),
            nn.Sigmoid()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        )

    def forward(self, x, res):
        xrc = torch.cat([x, res], dim=1).to(x)  # 合并通道
        xrs = self.out_conv(x + res)   # 加值
        attn = self.attn_conv(xrc)  # 压缩
        return xrs * attn

class GeoDecoder(nn.Module):
    def __init__(self,  norm_fn, acf, channel_lists=[256, 192, 128, 64, 32, 32]):
        super(GeoDecoder, self).__init__()
        if norm_fn == "batch":
            norm = nn.BatchNorm2d
        elif norm_fn == "instance":
            norm = nn.InstanceNorm2d
        else:
            raise Exception("wrong norm type!")
        self.norm_fn = norm_fn
        self.acf = acf

        self.attn = AttentionBlock(256, 32, 16, 256, norm, acf, type=0)

        self.in_planes = channel_lists[0]  # 来自特征的维度
        self.olayer1 = self._make_layer(channel_lists[1], stride=-2)

        self.attn2 = SpatialAttn(channel_lists[1])
        self.olayer2 = self._make_layer(channel_lists[2], stride=-2)

        self.attn3 = SpatialAttn(channel_lists[2], channel_lists[3])
        self.olayer3 = self._make_layer(channel_lists[3], stride=-2)

        self.attn4 = SpatialAttn(channel_lists[3], channel_lists[4])
        self.olayer4 = self._make_layer(channel_lists[4], stride=-2)

        self.se5 = SEModule(channel_lists[5])
        self.attn5 = SpatialAttn(channel_lists[5], channel_lists[5])

        self.post_p = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            norm(32),
            acf,
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0),
            nn.Hardtanh(-1, 1)
        )

    def _make_layer(self, dim, stride=1):
        layer1 = ResBlock(self.in_planes, dim, norm_fn=self.norm_fn, stride=stride, acf=self.acf)
        layer2 = ResBlock(dim, dim, norm_fn=self.norm_fn, stride=1, acf=self.acf)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x0, x1, x2, x3, f, x):

        # 坐标映射
        b, c, h, w = f.shape

        coords = gen_coords(h, w)[None].to(f)
        pn_embed = get_positional_embeddings(coords[:, 0], coords[:, 1], 32 // 2).repeat(b, 1, 1, 1)
        coords_feature = self.attn(pn_embed, f)

        x = coords_feature
        x = self.olayer1(x)
        x = self.olayer2(self.attn2(x, x3))
        x = self.olayer3(self.attn3(x, x2))
        x = self.olayer4(self.attn4(x, x1))
        # 组合
        f = self.attn5(self.se5(x), x0)
        # 上采样
        grid = self.post_p(f)

        return grid

class GeoRec(nn.Module):
    def __init__(self, in_channel=3, norm_fn="batch", acf=nn.LeakyReLU(inplace=True)):
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

    summary(model, (2, 3, 448, 448), depth=1, device="cpu")