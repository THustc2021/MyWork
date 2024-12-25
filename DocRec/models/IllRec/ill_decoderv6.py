import torch
from models.Extractor.encoder import *
from utils.common import reload_model

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
        xrs = self.out_conv(res)   # 加值
        attn = self.attn_conv(xrc)  # 压缩
        return xrs * attn

class IllDecoder(nn.Module):

    def __init__(self, norm_fn="batch", acf=nn.ReLU(inplace=True),
                 channel_lists=[256, 192, 128, 64, 32, 32], final_upsample=False):
        super(IllDecoder, self).__init__()

        self.norm_fn = norm_fn
        self.acf = acf

        self.in_planes = channel_lists[0]  # 来自特征的维度
        self.se1 = SEModule(channel_lists[0])
        self.olayer1 = self._make_layer(channel_lists[1], stride=-2)

        self.se2 = SEModule(channel_lists[1])
        self.attn2 = SpatialAttn(channel_lists[1])
        self.olayer2 = self._make_layer(channel_lists[2], stride=-2)

        self.se3 = SEModule(channel_lists[2])
        self.attn3 = SpatialAttn(channel_lists[2], channel_lists[3])
        self.olayer3 = self._make_layer(channel_lists[3], stride=-2)

        self.se4 = SEModule(channel_lists[3])
        self.attn4 = SpatialAttn(channel_lists[3], channel_lists[4])
        self.olayer4 = self._make_layer(channel_lists[4], stride=-2)

        self.se5 = SEModule(channel_lists[5])
        self.attn5 = SpatialAttn(channel_lists[5], channel_lists[5])

        if final_upsample:
            self.final_upsample = self._make_layer(channel_lists[5], stride=-2)
        else:
            self.final_upsample = None

        self.postprocess = nn.Sequential(
            nn.Conv2d(channel_lists[5], channel_lists[5], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_lists[5]),
            self.acf,
            nn.Conv2d(channel_lists[5], 3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def _make_layer(self, dim, stride=1):
        layer1 = ResBlock(self.in_planes, dim, norm_fn=self.norm_fn, stride=stride, acf=self.acf)
        layer2 = ResBlock(dim, dim, norm_fn=self.norm_fn, stride=1, acf=self.acf)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x0, x1, x2, x3, x):

        x = self.olayer1(self.se1(x))
        x = self.olayer2(self.attn2(self.se2(x), x3))
        x = self.olayer3(self.attn3(self.se3(x), x2))
        x = self.olayer4(self.attn4(self.se4(x), x1))
        # 暂时用不到
        if self.final_upsample != None:
            x = self.final_upsample(x)
        # 组合
        f = self.attn5(self.se5(x), x0)
        # 输出结果
        out = self.postprocess(f)

        return out

class IllRec(nn.Module):

    def __init__(self, in_channel=3, norm_fn="batch", acf=nn.ReLU(inplace=True)):
        super(IllRec, self).__init__()

        self.extractor = FeatureEncoder(in_channel, norm_fn, acf)
        self.ill_decoder = IllDecoder(norm_fn, acf)

    def load_model(self, state_dict, map_location="cuda", not_use_parrel_trained=True):
        reload_model(self, state_dict, map_device=map_location, not_use_parrel_trained=not_use_parrel_trained)

    def forward(self, inp):
        x0, x1, x2, x3, x = self.extractor(inp)
        out = self.ill_decoder(x0, x1, x2, x3, x)
        return out

if __name__ == '__main__':

    from torchinfo import summary
    summary(IllRec(), (2, 3, 224, 224), device="cpu", depth=1)

    total_num = sum(p.numel() for p in IllRec().parameters())
    trainable_num = sum(p.numel() for p in IllRec().parameters() if p.requires_grad)
    print(total_num, trainable_num)

    import torch
    from thop import profile
    model = IllRec()
    input = torch.randn(1, 3, 512, 512)
    Flops, params = profile(model, inputs=(input,))  # macs
    print('Flops: % .4fG' % (Flops / 1000000000))  # 计算量
    print('params参数量: % .4fM' % (params / 1000000))