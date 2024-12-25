import torch
import torch.nn.functional as F
from torch import nn

from models.Extractor.encoder import FeatureEncoder
from utils.common import reload_model
from models.GeoRec.basic.CAM import *

class JigsawGeoDecoder(nn.Module):
    def __init__(self,  cls_num, norm_fn, acf):
        super(JigsawGeoDecoder, self).__init__()
        if norm_fn == "batch":
            norm = nn.BatchNorm2d
        elif norm_fn == "instance":
            norm = nn.BatchNorm2d
        else:
            raise Exception("wrong norm type!")

        self.avgpool1 = nn.AvgPool2d((2, 2))
        self.attn1 = AttentionBlock(192, 256, 128, 192, norm=norm, acf=acf, type=0)
        self.add_and_norm1 = Add_and_Norm(192, acf)
        self.avgpool2 = nn.AvgPool2d((4, 4))
        self.attn2 = AttentionBlock(128, 192, 65, 128, norm=norm, acf=acf, type=0)
        self.add_and_norm2 = Add_and_Norm(128, acf)
        self.avgpool3 = nn.AvgPool2d((8, 8))
        self.attn3 = AttentionBlock(64, 128, 32, 64, norm=norm, acf=acf, type=0)
        self.add_and_norm3 =  Add_and_Norm(64, acf)
        self.avgpool4 = nn.AvgPool2d((16, 16))
        self.attn4 = AttentionBlock(32, 64, 32, 32, norm=norm, acf=acf, type=0)
        self.add_and_norm4 =  Add_and_Norm(32, acf)
        self.attn0 = AttentionBlock(32, 32, 32, 32, norm, acf)
        self.add_and_norm5 =  Add_and_Norm(32, acf)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.cls_head = nn.Sequential(
            nn.Linear(32, cls_num),
            # nn.Softmax(dim=1)
        )
        self.cls_num = cls_num
        # self.attn = AttentionBlock(32, 32, 32, 32, norm, acf, type=1)
        # self.upsample = Upsampler(32, 32, norm, acf, up_scale=16)
        #
        # self.post_p = nn.Sequential(
        #     nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        #     norm(32),
        #     acf,
        #     nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0),
        #     nn.Hardtanh(-1, 1)
        # )

    def forward(self, x0, x1, x2, x3, f, x):

        x3 = self.avgpool1(x3)
        f1 = self.add_and_norm1(self.attn1(f, x3), x3)
        x2 = self.avgpool2(x2)
        f2 = self.add_and_norm2(self.attn2(f1, x2), x2)
        x1 = self.avgpool3(x1)
        f3 = self.add_and_norm3(self.attn3(f2, x1), x1)
        x0 = self.avgpool4(x0)
        f4 = self.add_and_norm4(self.attn4(f3, x0), x0)
        f4 = self.add_and_norm5(self.attn0(f4, f4), f4)

        c = self.gap(f4).squeeze()
        c = self.cls_head(c)
        # 坐标映射
        # h, w = x.shape[2:]
        # coords = gen_coords(h//16, w//16)[None].to(f)
        # pn_embed = get_positional_embeddings(coords[:, 0], coords[:, 1], 32 // 2)
        # coords_embed = pn_embed.repeat(f.shape[0], 1, 1, 1).to(f)  # (b, c, h, w)
        # grid = self.attn(coords_embed, f4)
        # # 上采样
        # grid = self.upsample(grid)
        # grid = self.post_p(grid)

        return c

class JigsawGeoDecoderv2(nn.Module):
    def __init__(self,  cls_num, norm_fn, acf):
        super(JigsawGeoDecoderv2, self).__init__()
        if norm_fn == "batch":
            norm = nn.BatchNorm2d
        elif norm_fn == "instance":
            norm = nn.BatchNorm2d
        else:
            raise Exception("wrong norm type!")

        self.avgpool1 = nn.AvgPool2d((2, 2))
        self.attn1 = AttentionBlock(192, 256, 128, 192, norm=norm, acf=acf, type=0)
        self.add_and_norm1 = Add_and_Norm(192, acf)
        self.avgpool2 = nn.AvgPool2d((4, 4))
        self.attn2 = AttentionBlock(128, 192, 65, 128, norm=norm, acf=acf, type=0)
        self.add_and_norm2 = Add_and_Norm(128, acf)
        self.avgpool3 = nn.AvgPool2d((8, 8))
        self.attn3 = AttentionBlock(64, 128, 32, 64, norm=norm, acf=acf, type=0)
        self.add_and_norm3 =  Add_and_Norm(64, acf)
        self.avgpool4 = nn.AvgPool2d((16, 16))
        self.attn4 = AttentionBlock(32, 64, 32, 32, norm=norm, acf=acf, type=0)
        self.add_and_norm4 =  Add_and_Norm(32, acf)
        self.attn0 = AttentionBlock(32, 32, 32, 32, norm, acf)
        self.add_and_norm5 =  Add_and_Norm(32, acf)

        self.attn = AttentionBlock(32, 32, 32, 32, norm, acf, type=1)
        self.cls_num = cls_num
        # self.upsample = Upsampler(32, 32, norm, acf, up_scale=16)
        #
        self.post_p = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            norm(32),
            acf,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Softmax(dim=2)
        )

    def forward(self, x0, x1, x2, x3, f, x):

        x3 = self.avgpool1(x3)
        f1 = self.add_and_norm1(self.attn1(f, x3), x3)
        x2 = self.avgpool2(x2)
        f2 = self.add_and_norm2(self.attn2(f1, x2), x2)
        x1 = self.avgpool3(x1)
        f3 = self.add_and_norm3(self.attn3(f2, x1), x1)
        x0 = self.avgpool4(x0)
        f4 = self.add_and_norm4(self.attn4(f3, x0), x0)
        f4 = self.add_and_norm5(self.attn0(f4, f4), f4)

        # 坐标映射
        coords = gen_coords(self.cls_num, self.cls_num)[None].to(f)
        pn_embed = get_positional_embeddings(coords[:, 0], coords[:, 1], 32 // 2)
        coords_embed = pn_embed.repeat(f.shape[0], 1, 1, 1).to(f)  # (b, c, h, w)
        grid = self.attn(coords_embed, f4)
        # # 上采样
        # grid = self.upsample(grid)
        grid = self.post_p(grid).squeeze()

        return grid

class JigsawGeoRec(nn.Module):
    def __init__(self, cls_num, type, in_channel=3, norm_fn="batch", acf=nn.LeakyReLU()):
        super(JigsawGeoRec, self).__init__()
        self.norm_fn = norm_fn
        self.acf = acf
        self.extractor = FeatureEncoder(in_channel, norm_fn=norm_fn, acf=acf)
        if type == 0:
            self.geo_decoder = JigsawGeoDecoder(cls_num, norm_fn, self.acf)
        else:
            self.geo_decoder = JigsawGeoDecoderv2(cls_num, norm_fn, self.acf)

    def load_model(self, state_dict, map_location="cuda", not_use_parrel_trained=True):
        reload_model(self, state_dict, map_device=map_location, not_use_parrel_trained=not_use_parrel_trained)

    def forward(self, x):
        x0, x1, x2, x3, f = self.extractor(x)
        pred = self.geo_decoder(x0, x1, x2, x3, f, x)
        return pred


if __name__ == '__main__':
    model = JigsawGeoRec(cls_num=16)
    from torchinfo import summary

    summary(model, (2, 3, 224, 224), depth=1, device="cpu")