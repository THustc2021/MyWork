import torch
from torchvision.models.segmentation import DeepLabV3

from models.IllRec.ill_decoderv7 import *
from models.GeoRec.geo_decoder_lwv20 import *
from utils.common import reload_model


class UnifyNet(nn.Module):
    """
        同时学习光照和变形信息
    """

    def __init__(self, in_channel=3, norm_fn="batch", acf=nn.ReLU(inplace=True)):
        super(UnifyNet, self).__init__()

        self.norm_fn = norm_fn
        self.acf = acf

        self.extractor = FeatureEncoder(in_channel, norm_fn, acf)  # (bs, 512, h, w)

        self.ill_decoder = IllDecoder(norm_fn, acf)
        self.geo_decoder = GeoDecoder(norm_fn, acf)

    def load_model(self, state_dict, map_location="cuda", not_use_parrel_trained=True):
        reload_model(self, state_dict, map_device=map_location, not_use_parrel_trained=not_use_parrel_trained)

    def forward_ill(self):
        ill_out = self.ill_decoder(*self.features[:-1])
        return ill_out

    def forward_geo(self):
        out = self.geo_decoder(*self.features)
        return out

    def forward_set_feature(self, x):
        x0, x1, x2, x3, f = self.extractor(x)
        self.features = [x0, x1, x2, x3, f, x]

    def forward(self, x, return_final_result=False):

        x0, x1, x2, x3, f = self.extractor(x)
        ill_out = self.ill_decoder(x0, x1, x2, x3, f)
        geo_out = self.geo_decoder(x0, x1, x2, x3, f, x)

        if return_final_result:  # 返回几何矫正和光照修正结果
            import torch.nn.functional as F
            res = F.grid_sample(ill_out, geo_out.permute(0, 2, 3, 1), align_corners=False)
            return res

        return ill_out, geo_out

if __name__ == '__main__':

    from torchinfo import summary
    summary(UnifyNet(), (2, 3, 224, 224), device="cpu")

    # import time
    # import numpy as np
    # with torch.no_grad():
    #     for _ in range(100):
    #         UnifyNet().cuda()(torch.rand(2, 3, 448, 448).cuda())
    #     times = []
    #     seg_model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=False).cuda()
    #     model = UnifyNet().cuda()
    #     for _ in range(1000):
    #         s = time.time()
    #         seg_model(torch.rand(2, 3, 256, 256).cuda())
    #         model(torch.rand(2, 3, 448, 448).cuda())
    #         torch.cuda.synchronize()
    #         e = time.time()
    #         times.append(e-s)
    #     print(f"time: {np.mean(times)}")

    total_num = sum(p.numel() for p in UnifyNet().parameters())
    trainable_num = sum(p.numel() for p in UnifyNet().parameters() if p.requires_grad)
    print(total_num, trainable_num)


    # from thop import profile
    #
    # input = torch.randn(1, 3, 224, 224)
    # flops, params = profile(UnifyNet(), inputs=(input,))