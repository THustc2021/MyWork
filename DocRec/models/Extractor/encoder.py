from torch import nn

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, norm_fn="instance", acf=nn.LeakyReLU(inplace=True),
                 dilation=1, return_branch=False):
        super(ResBlock, self).__init__()

        if norm_fn == "batch":
            norm = nn.BatchNorm2d
        elif norm_fn == "instance":
            norm = nn.InstanceNorm2d
        else:
            raise "Unsupported norm type!"
        self.acf = acf

        self.norm1 = norm(out_channels)
        if stride > 0:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation)
        else:   # 不能为-1和0
            self.conv1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=-stride, padding=1, output_padding=1)
        self.norm2 = norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        if stride > 0:
            self.sample = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)    # 无论有没有采样，都要重整通道
        else:
            self.sample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=-stride, stride=-stride)

        self.return_branch = return_branch

    def forward(self, x):

        if isinstance(x, tuple):
            x, res = x
        else:
            res = None

        residual = x
        residual = self.sample(residual)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.acf(out)

        out = self.conv2(out)
        out = self.acf(self.norm2(out + residual))

        if res != None:
            return out, res
        if self.return_branch:
            return out, residual

        return out

class FeatureEncoder(nn.Module):

    def __init__(self, in_channel=3, norm_fn="instance", acf=nn.ReLU(inplace=True)):
        super(FeatureEncoder, self).__init__()

        self.norm_fn = norm_fn
        self.acf = acf

        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            self.acf,
            nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        )

        self.in_planes = 32
        self.layer1 = self._make_layer(64, stride=2, dilation=2, return_branch=True)
        self.layer2 = self._make_layer(128, stride=2, dilation=2, return_branch=True)   # 大核滤去高频细节
        self.layer3 = self._make_layer(192, stride=2, dilation=2, return_branch=True)
        self.layer4 = self._make_layer(256, stride=2)

    def _make_layer(self, dim, stride=1, **kwargs):
        layer1 = ResBlock(self.in_planes, dim, norm_fn=self.norm_fn, stride=stride, acf=self.acf, **kwargs)
        layer2 = ResBlock(dim, dim, norm_fn=self.norm_fn, stride=1, acf=self.acf, **kwargs)
        layers = (layer1, layer2)

        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, inp):
        x0 = self.preprocess(inp)
        x, x1 = self.layer1(x0)   # (bs, 32, h, w)
        x, x2 = self.layer2(x)    # (bs, 64
        x, x3 = self.layer3(x)    # (bs, 128
        x = self.layer4(x)    # (bs, 256

        return x0, x1, x2, x3, x

if __name__ == '__main__':
    from torchinfo import summary
    summary(FeatureEncoder(), (2, 3, 224, 224))