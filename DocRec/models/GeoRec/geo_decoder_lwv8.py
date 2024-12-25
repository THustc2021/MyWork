import math

import torch
import torch.nn.functional as F

from models.Extractor.encoder import FeatureEncoder
from models.GeoRec.basic.CAM import *
from utils.common import reload_model
from torch import nn

convs_setting = {
    1: [3, 1, 1, 1],
    2: [3, 2, 1, 1],
    4: [3, 4, 1, 2],
    8: [5, 8, 1, 2],
    16: [7, 16, 3, 3]
}

class SpatialAttn(nn.Module):
    def __init__(self, in_channel, downsample_rate):
        super(SpatialAttn, self).__init__()

        mc = in_channel
        self.conv = nn.Conv2d(in_channel, mc, *convs_setting[downsample_rate])
        self.avgpool = nn.MaxPool2d((downsample_rate, downsample_rate))
        self.attn = nn.Sequential(
            nn.Conv2d(in_channel+mc, in_channel+mc, 3, 1, 1),
            nn.BatchNorm2d(in_channel+mc),
            nn.ReLU(),
            nn.Conv2d(in_channel+mc, 1, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):

        pool_x = self.avgpool(x)
        conv_x = self.conv(x)    # 采样特征

        y = torch.cat([pool_x, conv_x], dim=1).to(x)
        attn = self.attn(y)

        return pool_x * attn

class GeoDecoder(nn.Module):
    def __init__(self,  norm_fn, acf):
        super(GeoDecoder, self).__init__()
        if norm_fn == "batch":
            norm = nn.BatchNorm2d
        elif norm_fn == "instance":
            norm = nn.InstanceNorm2d
        else:
            raise Exception("wrong norm type!")

        self.attn = AttentionBlock(256, 32, 8, 32, norm, acf, type=0)

        self.spattn0 = SpatialAttn(32, downsample_rate=16)
        self.attn_hq_with_coords0 = AttentionBlock(32, 32, 8, 64, norm, acf, type=2)
        #
        self.spattn1 = SpatialAttn(64, downsample_rate=8)
        self.attn_hq_with_coords1 = AttentionBlock(64, 64, 8, 128, norm, acf, type=2)
        #
        self.spattn2 = SpatialAttn(128, downsample_rate=4)
        self.attn_hq_with_coords2 = AttentionBlock(128, 128, 8, 192, norm, acf, type=2)
        #
        self.spattn3 = SpatialAttn(192, downsample_rate=2)
        self.attn_hq_with_coords3 = AttentionBlock(192, 192, 8, 256, norm, acf, type=2)

        self.upsample = Upsampler(256, 32, norm, acf, up_scale=16)
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

        # hq_feature0 = F.max_pool2d(x0, (16, 16))
        hq_feature0 = self.spattn0(x0)
        coords_feature = self.attn_hq_with_coords0(hq_feature0, coords_feature)
        #
        # hq_feature1 = F.max_pool2d(x1, (8, 8))
        hq_feature1 = self.spattn1(x1)
        coords_feature = self.attn_hq_with_coords1(hq_feature1, coords_feature)

        #
        # hq_feature2 = F.max_pool2d(x2, (4, 4))
        hq_feature2 = self.spattn2(x2)
        coords_feature = self.attn_hq_with_coords2(hq_feature2, coords_feature)

        #
        # hq_feature3 = F.max_pool2d(x3, (2, 2))
        hq_feature3 = self.spattn3(x3)
        coords_feature = self.attn_hq_with_coords3(hq_feature3, coords_feature)

        # 上采样
        grid = self.upsample(coords_feature)
        grid = self.post_p(grid)

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