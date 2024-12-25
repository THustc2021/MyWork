from models.Extractor.encoder import *
from utils.common import reload_model

class IllDecoder(nn.Module):

    def __init__(self, norm_fn="batch", acf=nn.ReLU(inplace=True),
                 channel_lists=[256, 192, 128, 64, 32, 32], final_upsample=False):
        super(IllDecoder, self).__init__()

        self.norm_fn = norm_fn
        self.acf = acf

        self.in_planes = channel_lists[0]  # 来自特征的维度
        self.olayers1 = self._make_layer(channel_lists[1], stride=-2)
        self.olayers2 = self._make_layer(channel_lists[2], stride=-2)
        self.olayers3 = self._make_layer(channel_lists[3], stride=-2)
        self.olayers4 = self._make_layer(channel_lists[4], stride=-2)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.channel_conv = nn.Sequential(
            nn.ConvTranspose2d(channel_lists[4], channel_lists[5], kernel_size=3, stride=2, padding=1, output_padding=1)  if final_upsample else \
                nn.Conv2d(channel_lists[5], channel_lists[5], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_lists[5]),
            self.acf,
            nn.Conv2d(channel_lists[5], channel_lists[5], kernel_size=1, stride=1, padding=0)
        )       # 额外增加了一个
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

        x = self.olayers1(x) + self.gap(x3)
        x = self.olayers2(x) + self.gap(x2)
        x = self.olayers3(x) + self.gap(x1)
        x = self.olayers4(x)
        if self.final_upsample != None:
            x = self.final_upsample(x)
        # 组合
        f = self.channel_conv(x0) + x
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