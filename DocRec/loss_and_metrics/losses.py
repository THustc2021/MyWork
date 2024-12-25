import torch
import torch.nn.functional as F
from torch import nn

from loss_and_metrics.vggloss import VGGLoss


class IllLoss(nn.Module):
    def __init__(self, path, base_loss=nn.L1Loss, alpha=0.5):
        super(IllLoss, self).__init__()

        self.vggloss = VGGLoss(path)
        self.baseloss = base_loss()

        self.alpha = alpha

    def forward(self, x, y):

        vgg_loss = self.vggloss(x, y)
        base_loss = self.baseloss(x, y)

        return self.alpha * vgg_loss + base_loss

def boundary_loss(input, target):
    # 边界预测到文档内部加大惩罚，预测到文档外部随距离线性增长
    def acf(dis):
        dis[dis >= 0] = torch.exp(dis[dis >= 0]) - 1    # 在内部
        dis[dis < 0] = -dis[dis<0]  # 在外部
        return dis
    u = acf(input[:, :, 0, :]-target[:, :, 0, :]).mean()
    d = acf(-input[:, :, -1, :]+target[:, :, -1, :]).mean()
    l = acf(input[:, :, 0, :]-target[:, :, 0, :]).mean()
    r = acf(-input[:, :, -1, :]+target[:, :, -1, :]).mean()
    return u+d+l+r

class MLocal_loss(nn.Module):

    def __init__(self, max_size, permit_dilation=True, device="cuda"):
        super(MLocal_loss, self).__init__()

        self.main_loss = nn.L1Loss()
        # 初始化各项损失滤波器
        # 变形识别以及局部损失核
        filters = []
        for size in range(3, max_size + 1, 2):
            sn = size * size
            filter = torch.eye(sn, sn)
            filter[:, sn // 2] -= 1
            filter = filter.view(sn, size, size)
            filter = torch.cat([filter[:(sn // 2)], filter[(sn // 2 + 1):]], dim=0).to(device)
            filters.append(filter)
        self.filters = filters
        self.permit_dilation = permit_dilation

    def forward(self, output, label):
        # ms local loss
        local_losses = []
        fi = 0
        for f in self.filters:
            # 两组替换流的局部相似性
            filter = f[:, None].to(output)
            # 折叠两组坐标
            ori_grid_x = output[:, 0][:, None]
            ori_grid_y = output[:, 1][:, None]
            ori_x_feat = F.conv2d(ori_grid_x, filter, stride=1, padding=0)
            ori_y_feat = F.conv2d(ori_grid_y, filter, stride=1, padding=0)
            ori_feat = torch.stack([ori_x_feat, ori_y_feat], dim=1).to(output)

            tgt_grid_x = label[:, 0][:, None]
            tgt_grid_y = label[:, 1][:, None]
            tgt_x_feat = F.conv2d(tgt_grid_x, filter, stride=1, padding=0)
            tgt_y_feat = F.conv2d(tgt_grid_y, filter, stride=1, padding=0)
            tgt_feat = torch.stack([tgt_x_feat, tgt_y_feat], dim=1).to(label)

            loss_both = F.l1_loss(ori_feat, tgt_feat)

            fi += 1

            if self.permit_dilation:
                ori_x_feat = F.conv2d(ori_grid_x, filter, stride=1, padding=0, dilation=2)
                ori_y_feat = F.conv2d(ori_grid_y, filter, stride=1, padding=0, dilation=2)
                ori_feat = torch.stack([ori_x_feat, ori_y_feat], dim=1).to(output)
                tgt_x_feat = F.conv2d(tgt_grid_x, filter, stride=1, padding=0, dilation=2)
                tgt_y_feat = F.conv2d(tgt_grid_y, filter, stride=1, padding=0, dilation=2)
                tgt_feat = torch.stack([tgt_x_feat, tgt_y_feat], dim=1).to(label)
                loss_both += F.l1_loss(ori_feat, tgt_feat)

                fi += 1

            local_losses.append(loss_both)

        local_losses = torch.mean(torch.stack(local_losses, dim=0))
        return local_losses
    
class GeoLoss(nn.Module):
    
    def __init__(self, w_bnd=0.5, w_local=0.5):
        super(GeoLoss, self).__init__()

        self.l1_loss = nn.L1Loss()
        self.local_loss = MLocal_loss(5)

        self.w_bnd = w_bnd
        self.w_local = w_local

    def forward(self, x, y):
        l1_loss = self.l1_loss(x, y)
        bnd_loss = boundary_loss(x, y)
        mlocal_loss = self.local_loss(x, y)

        total_loss = l1_loss + self.w_bnd*bnd_loss + self.w_local * mlocal_loss
        return total_loss

