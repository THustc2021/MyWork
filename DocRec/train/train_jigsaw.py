import datetime
import albumentations as A
import segmentation_models_pytorch
import torch
import torchvision.transforms
import vit_pytorch
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler, SGD
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms
from tqdm import tqdm

from dataprocess.Doc3D.data_jigsaw import *
from loss_and_metrics.losses import IllLoss
from loss_and_metrics.metric import SSIM
from models.Contrasive.GeoRec import *
from models.Contrasive.IllRec import ContrasiveIllRec
from loss_and_metrics.contrastiveloss import info_nce_loss
from utils.common import *
from dataprocess.Doc3D import data_Doc3D, dataset_docunet_valid

class CEWithMask(nn.Module):

    def __init__(self):
        super(CEWithMask, self).__init__()

        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, label):
        # if msk_idx is None:
        #     ce_loss = self.ce(pred[ne_idx], label[ne_idx])
        #     acc = torch.sum(pred[ne_idx].argmax(dim=1) == label[ne_idx]) / (label.shape[0] * label.shape[1])  # 带ne的acc
        #     return ce_loss, acc
        # else:
        #     final_idx = ~msk_idx & ne_idx   # 非msk以及非空
        #     msk_label = label[final_idx]
        #     msk_pred = pred[final_idx]
        #     # final_loss = self.ce(msk_pred, msk_label)
        #     final_loss = self.ce(pred, label)
        #     return final_loss, torch.sum(pred[final_idx].argmax(dim=1) == label[final_idx]) / final_idx.sum()
        ce_loss = self.ce(pred, label)
        acc = torch.sum(pred.argmax(dim=1) == label) / label.shape[0]
        return ce_loss, acc

# class CEWithMaskv2(nn.Module):
#     # 或许应对位置的标签和标签的位置两个分别求CE
#     def __init__(self):
#         super(CEWithMaskv2, self).__init__()
#
#     def forward(self, pred, label, ne_idx, msk_idx=None):
#         if msk_idx is None:
#             ce_loss1 = F.cross_entropy(pred, label, ignore_index=ne_idx)
#             acc = torch.sum(pred[ne_idx].argmax(dim=1) == label[ne_idx]) / (label.shape[0] * label.shape[1])  # 带ne的acc
#             return ce_loss, acc
#         else:
#             final_idx = ~msk_idx & ne_idx   # 非msk以及非空
#             msk_label = label[final_idx]
#             msk_pred = pred[final_idx]
#             final_loss = self.ce(msk_pred, msk_label)
#             return final_loss, torch.sum(pred[final_idx].argmax(dim=1) == label[final_idx]) / final_idx.sum()

def train(idx, dataloader, model, writer, loss_f, cuda=True):
    model.train()

    scaler = GradScaler()
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

        img_n, labels, img = data # 这样构造，可以满足loss的结构
        if cuda:
            img_n, labels, img = img_n.cuda(), labels.cuda(), img.cuda()

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda' if cuda else 'cpu', dtype=torch.float16):
            # 前向传播
            pred = model(img_n)
            # 光照损失
            final_loss, acc = loss_f(pred, labels)

        # 反向传播
        scaler.scale(final_loss).backward()
        # 更新模型参数
        scaler.step(optimizer)
        scaler.update()

        # 打印损失
        if idx % 10 == 0:
            writer.add_scalar("Train/loss", final_loss.item(), idx)
            writer.add_scalar("Train/acc", acc.item(), idx)
            writer.add_scalar("Train/lr", optimizer.state_dict()['param_groups'][0]['lr'], idx)

            scheduler.step(final_loss)

        # 更新步骤计数器
        idx += 1

    # if idx % 1000 == 0:
    #     inp_n = img_n.float().detach().cpu().numpy().transpose(0, 2, 3, 1)
    #     inp = img.float().detach().cpu().numpy().transpose(0, 2, 3, 1)
    #     fig, ax = plt.subplots(3, 2, dpi=100)
    #     for i in range(3):
    #         ax[i, 0].axis(False)
    #         ax[i, 0].title.set_text("img transform, size: {}".format(inp_n[i].shape))
    #         ax[i, 0].imshow(inp_n[i])
    #         ax[i, 1].axis(False)
    #         ax[i, 1].title.set_text("img origin, size: {}".format(inp[i].shape))
    #         ax[i, 1].imshow(inp[i])
    #     writer.add_figure(f"Fig/{idx}", fig)
    #     writer.flush()

    return idx


if __name__ == '__main__':

    set_determistic()  # 可复现性

    train_dirs = [
                  # "/home/xtanghao/THPycharm/dataset/Doc3D/ill_gt",
                  "/home/xtanghao/THPycharm/dataset/DocUnet Benchmark/scan",
                  "/home/xtanghao/THPycharm/dataset/DIR300/gt",
                  "/home/xtanghao/THPycharm/dataset/DocReal/scanned",
                  "/home/xtanghao/THPycharm/dataset/WarpDoc/digital",
                  # "/home/xtanghao/THPycharm/dataset/WarpDoc/digital_margin",
                  "/home/xtanghao/THPycharm/dataset/UVDoc_final/textures",
                  # "/home/xtanghao/THPycharm/dataset/docs-sm"
                  ]
    vgg_path = "../model_pretrained/vgg19.pkl"

    train_bs = 48
    valid_bs = 48
    num_workers = 8

    device = "cuda"

    lr = 1e-3
    factor = 0.8
    wd = 1e-4
    patience = 5
    min_lr = 1e-6
    epochs = 50

    img_shape = (300, 300)
    patch_shape = (150, 150)
    crop_transform = A.Compose(
        [
            # A.RandomResizedCrop(height=448, width=448),
            # A.PadIfNeeded(min_height=448, min_width=448),
            # A.RandomCrop(height=448, width=448),
            A.Resize(height=448, width=448),
            # A.OneOf([
            #     A.ColorJitter(),
            #     A.GaussianBlur(),
            #     A.ISONoise(),
            #     A.Emboss(),
            # ], p=0.1),
            # A.OneOf([
            #     A.PixelDropout(),
            #     A.Defocus(),
            #     A.CLAHE()
            # ], p=0.1),
            ToTensorV2(transpose_mask=True)
         ]
    )
    train_transform = None
    #     torchvision.transforms.Compose([
    #     transforms.RandomChoice([
    #         transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=(0.2, 0.2)),
    #         transforms.RandomPerspective()
    #     ])
    # ])

    train_dataset = JigsawDatasetv2(train_dirs,
                                  patch_shape=patch_shape,
                                  img_shape=img_shape,
                                  crop_transform=crop_transform,
                                  img_transform=train_transform,
                                  permute_path="/home/xtanghao/DocRec/bin/permutations/permutations_hamming_max_100.npy")

    train_dataloader = DataLoader(train_dataset, batch_size=train_bs, drop_last=False, shuffle=True,
                                  num_workers=num_workers)

    model = JigsawGeoRec(24, type=0).to(device)
    # model = vit_pytorch.vit.ViT(image_size=(448, 448), patch_size=(16, 16), num_classes=10, dim=1024, heads=8, depth=5, mlp_dim=1024).to(device)

    log_dir = f"../results/{model._get_name()}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}"
    writer = SummaryWriter(log_dir=log_dir)

    # optimizer = AdamW(model.parameters(), lr=lr, amsgrad=False)
    optimizer = SGD(model.parameters())
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr,
                                               verbose=True)
    loss_f = CEWithMask()

    # 记录超参数信息
    writer.add_hparams({
        "train_bs": train_bs, "valid_bs": valid_bs, "num_workers": num_workers,
        "lr": lr, "factor": factor, "wd": wd, "patience": patience, "min_lr": min_lr, "epochs": epochs,
        "img_shape": str(img_shape), "train_transform": str(train_transform),
        "model": str(model), "optimizer": str(optimizer), "scheduler": str(scheduler),
        "ill_loss_f": str(loss_f)
    }, {}, run_name="experiment setting")
    # # 记录模型静态图
    # writer.add_graph(model, torch.rand((2, 3, 448, 448)).cuda(), verbose=True)
    # writer.flush()
    writer.add_text("Desc", "V0 No Coords")

    idx = 0
    for epoch in range(epochs):
        idx = train(idx, train_dataloader, model, writer, loss_f)
        # 保存模型
        torch.save(model.state_dict(), os.path.join(log_dir, "model_{}.pth").format(epoch))

        # 重置
        if (epoch+1) % 20 == 0:
            optimizer = AdamW(model.parameters(), lr=lr, amsgrad=True)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience,
                                                   min_lr=min_lr,
                                                   verbose=True)
            print(f"Epoch {epoch} finished.")

    print("train done. save model parameters done.")
    writer.close()
