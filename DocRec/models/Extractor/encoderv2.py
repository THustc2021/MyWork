from torch import nn
import segmentation_models_pytorch as smp

class FeatureEncoder(nn.Module):

    def __init__(self, in_channel=3, norm_fn="instance", acf=nn.ReLU(inplace=True)):
        super(FeatureEncoder, self).__init__()

        # self.norm_fn = norm_fn
        # self.acf = acf

        # self.preprocess = nn.Sequential(
        #     nn.Conv2d(in_channel, 32, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(32),
        #     self.acf,
        #     nn.Conv2d(32, 32, kernel_size=1, stride=1, padding=0)
        # )
        #
        # self.in_planes = 32
        # self.layer1 = self._make_layer(64, stride=2, dilation=2, return_branch=True)
        # self.layer2 = self._make_layer(128, stride=2, dilation=2, return_branch=True)   # 大核滤去高频细节
        # self.layer3 = self._make_layer(192, stride=2, dilation=2, return_branch=True)
        # self.layer4 = self._make_layer(256, stride=2)
        self.extractor = smp.encoders.get_encoder("resnet34")

    def forward(self, inp):
        x = self.extractor(inp)

        return x

if __name__ == '__main__':

    from torchsummary import summary

    summary(FeatureEncoder().cuda(), (3, 448, 448))