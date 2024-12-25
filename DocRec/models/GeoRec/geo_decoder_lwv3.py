import torch
import torch.nn.functional as F

from models.Extractor.encoder import FeatureEncoder
from utils.common import reload_model
from torch import nn


def gen_coords(h, w):
    coords = torch.meshgrid(torch.arange(h), torch.arange(w))
    coords = torch.stack([2 * coords[1] / (w - 1) - 1,
                          2 * coords[0] / (h - 1) - 1], dim=0).float()  # 归一化坐标
    return coords  # [2, h, w]


class TAM(nn.Module):

    def __init__(self, inp_t_dim, inp_c_dim, tc_dim, out_dim, norm, acf, skip_connect=True):
        super(TAM, self).__init__()

        self.t_conv = nn.Sequential(
            nn.Conv2d(inp_t_dim, tc_dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            norm(tc_dim),
            nn.Tanh()
        )
        self.c_conv = nn.Sequential(
            nn.Conv2d(inp_c_dim, tc_dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            norm(tc_dim),
            nn.Tanh()
        )
        self.cv_conv = nn.Sequential(
            nn.Conv2d(inp_c_dim, out_dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            norm(out_dim),
            acf,
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            norm(out_dim),
            acf,
            nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        )

        # coords_res
        if skip_connect:
            self.skip_connect = nn.Sequential(
                nn.Conv2d(inp_c_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                norm(out_dim),
                acf,
                nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
            )  # no upsample
            self.skip_post = nn.Sequential(
                norm(out_dim),
                nn.Tanh()
            )
        else:
            self.skip_connect = None

        self.post_conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            norm(out_dim),
            acf,
            nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.post_post = nn.Sequential(
            norm(out_dim),
            nn.Tanh()
        )

    def forward(self, c_feat, t_feat):
        # c_feat: b, cc, ch, cw
        # t_feat, b, dc, dh, dw
        b, _, dh, dw = t_feat.shape

        # qkv
        t = self.t_conv(t_feat).flatten(2, 3).transpose(1, 2)
        c = self.c_conv(c_feat).flatten(2, 3)
        cv = self.cv_conv(c_feat).flatten(2, 3).transpose(1, 2)

        # cross attn
        attn_w = torch.bmm(t, c)
        attn_w = torch.softmax(attn_w, dim=1)

        # with v
        attn_v = torch.bmm(attn_w, cv)
        attn_v = attn_v.transpose(1, 2).view(b, -1, dh, dw)

        # skip connect
        out = attn_v
        if self.skip_connect != None:
            c_feat_up = self.skip_connect(c_feat)
            c_feat_up = F.interpolate(c_feat_up, (dh, dw), mode="bilinear")
            out = self.skip_post(out + c_feat_up)
        # out
        out = self.post_conv(out) + out
        out = self.post_post(out)

        return out


class TAMv2(nn.Module):

    def __init__(self, inp_t_dim, inp_c_dim, tc_dim, out_dim, norm, acf, skip_connect=True):
        super(TAMv2, self).__init__()

        self.t_conv = nn.Sequential(
            nn.Conv2d(inp_t_dim, tc_dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            norm(tc_dim),
            nn.Tanh()
        )
        self.c_conv = nn.Sequential(
            nn.Conv2d(inp_c_dim, tc_dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            norm(tc_dim),
            nn.Tanh()
        )
        self.tv_conv = nn.Sequential(
            nn.Conv2d(inp_t_dim, out_dim, kernel_size=3, stride=1, padding=2, dilation=2, bias=False),
            norm(out_dim),
            acf,
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            norm(out_dim),
            acf,
            nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )

        # coords_res
        if skip_connect:
            self.skip_connect = nn.Sequential(
                nn.Conv2d(inp_c_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
                norm(out_dim),
                acf,
                nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False)
            )
            self.skip_post = nn.Sequential(
                norm(out_dim),
                nn.Tanh()
            )
        else:
            self.skip_connect = None

        self.post_conv = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1, bias=False),
            norm(out_dim),
            acf,
            nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.post_post = nn.Sequential(
            norm(out_dim),
            nn.Tanh()
        )

    def forward(self, c_feat, t_feat):
        # t_feat: b, cc, ch, cw
        # c_feat, b, dc, dh, dw
        b, _, dh, dw = c_feat.shape

        # qkv
        c = self.c_conv(c_feat).flatten(2, 3).transpose(1, 2)
        t = self.t_conv(t_feat).flatten(2, 3)
        tv = self.tv_conv(t_feat).flatten(2, 3).transpose(1, 2)

        # cross attn
        attn_w = torch.bmm(c, t)
        attn_w = torch.softmax(attn_w, dim=1)

        # with v
        attn_v = torch.bmm(attn_w, tv)
        attn_v = attn_v.transpose(1, 2).view(b, -1, dh, dw)

        # skip connect
        # out
        out = attn_v
        if self.skip_connect != None:
            c_feat_up = self.skip_connect(c_feat)
            out = self.skip_post(out + c_feat_up)
        out = self.post_conv(out) + out
        out = self.post_post(out)

        return out


class AttentionBlock(nn.Module):
    def __init__(self, inp_tran_dim, inp_coords_dim, qk_dim, out_dim, norm, acf, type=0):
        super(AttentionBlock, self).__init__()

        if type == 0:
            self.tam = TAM(inp_tran_dim, inp_coords_dim, qk_dim, out_dim, norm, acf)  # upsample first time
        else:
            self.tam = TAMv2(inp_tran_dim, inp_coords_dim, qk_dim, out_dim, norm, acf)
        self.self_attn = TAMv2(out_dim, out_dim, qk_dim, out_dim, norm, acf)

    def forward(self, feat1, feat2):
        f_embed = self.tam(feat1, feat2)
        f_embed = self.self_attn(f_embed, f_embed)
        return f_embed

def get_positional_embeddings(x_embed, y_embed, num_pos_feats, temperature=10000):
    # [b, h, w]
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats).to(x_embed)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_y = y_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
    pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
    # print(pos.shape)
    return pos  # [b, num_pos_feats, h, w]

class Upsamplerx16(nn.Module):

    def __init__(self, inp_dim, out_dim, norm, acf, up_scale=16):
        super(Upsamplerx16, self).__init__()

        self.flowhead = nn.Sequential(
            nn.Conv2d(inp_dim, out_dim, kernel_size=3, stride=1, padding=1),
            norm(out_dim),
            acf,
            nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        )
        self.mask = nn.Sequential(
            nn.Conv2d(inp_dim, up_scale*up_scale, kernel_size=3, stride=1, padding=1),
            norm(up_scale*up_scale),
            acf,
            nn.Conv2d(up_scale*up_scale, up_scale*up_scale * 9, kernel_size=1, stride=1, padding=0)
        )
        self.up_scale = up_scale

    def forward(self, img_f):

        flow = self.flowhead(img_f)
        mask = self.mask(img_f) * .25

        N, C, H, W = flow.shape
        mask = mask.view(N, 1, 9, self.up_scale, self.up_scale, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)  # (N, 2, 16, 16, H, W)，使用mask在flow的3*3邻域内进行加权结合
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)

        return up_flow.reshape(N, C, self.up_scale * H, self.up_scale * W)

class Add_and_Norm(nn.Module):
    def __init__(self, in_dim, acf):
        super(Add_and_Norm, self).__init__()

        self.norm1 = nn.BatchNorm2d(in_dim)

        self.feed_forward = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_dim),
            acf,
            nn.Conv2d(in_dim, in_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(in_dim),
            acf
        )
        self.norm2 = nn.BatchNorm2d(in_dim)
        self.acf = acf

    def forward(self, x1, x2):
        x = self.acf(self.norm1(x1 + x2))
        x = self.feed_forward(x) + x
        x = self.acf(self.norm2(x))
        return x

class GeoDecoder(nn.Module):
    def __init__(self,  norm_fn, acf):
        super(GeoDecoder, self).__init__()
        if norm_fn == "batch":
            norm = nn.BatchNorm2d
        elif norm_fn == "instance":
            norm = nn.BatchNorm2d
        else:
            raise Exception("wrong norm type!")

        self.attn = AttentionBlock(256, 256, 128, 256, norm, acf, type=0)
        self.upsample = Upsamplerx16(256, 32, norm, acf)
        # self.upsample = nn.Sequential(
        #     # Upsampler(32, 32, norm, acf),
        #     # Upsampler(32, 32, norm, acf),
        #     Upsampler(128, 64, norm, acf),
        #     Upsampler(64, 32, norm, acf)
        # )
        self.post_p = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            norm(32),
            acf,
            nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0),
            nn.Hardtanh(-1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias != None:
                    nn.init.zeros_(m.bias)

    def forward(self, x0, x1, x2, x3, f, x):

        # 坐标映射
        # b, c, h, w = f.shape

        # coords = gen_coords(h, w)[None].to(f)
        # pn_embed = get_positional_embeddings(coords[:, 0], coords[:, 1], c // 2).repeat(b, 1, 1, 1)
        # grid = self.attn(pn_embed, f)

        # 上采样
        # grid = self.upsample(grid)
        grid = self.upsample(f)
        grid = self.post_p(grid)

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

    summary(model, (2, 3, 448, 448), depth=1, device="cpu")