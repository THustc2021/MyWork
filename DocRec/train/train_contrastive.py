import datetime
import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataprocess.Doc3D.data_contrastive import ContrastiveDatasetv2
from loss_and_metrics.losses import IllLoss
from loss_and_metrics.metric import SSIM
from models.Contrasive.IllRec import ContrasiveIllRec
from loss_and_metrics.contrastiveloss import info_nce_loss
from utils.common import *
from dataprocess.Doc3D import data_Doc3D, dataset_docunet_valid


def train(idx, dataloader, model, writer, loss_f, cuda=True):
    model.train()

    scaler = GradScaler()
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

        sample_t0, sample_t1, sample_t10, sample_t01 = data # 这样构造，可以满足loss的结构
        img = torch.cat([sample_t0, sample_t1, sample_t10, sample_t01], dim=0)
        if cuda:
            img = img.cuda()

        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda' if cuda else 'cpu', dtype=torch.float16):
            # 前向传播
            pred = model(img)
            # 光照损失
            iloss, floss = loss_f(pred)
            loss = iloss + floss

        # 反向传播
        scaler.scale(loss).backward()
        # 更新模型参数
        scaler.step(optimizer)
        scaler.update()

        # 打印损失
        if idx % 10 == 0:
            writer.add_scalar("Train/loss", loss.item(), idx)
            writer.add_scalar("Train/iloss", iloss.item(), idx)
            writer.add_scalar("Train/floss", floss.item(), idx)
            writer.add_scalar("Train/lr", optimizer.state_dict()['param_groups'][0]['lr'], idx)

            scheduler.step(loss)

        # 更新步骤计数器
        idx += 1

    # if save_result != None:
    #     save_result(img.float(), label.float(), ill_out.float())

    return idx

class ContrasiveLoss(nn.Module):
    
    def __init__(self):
        super(ContrasiveLoss, self).__init__()

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, fpred):

        inner_featurep, fp = fpred

        # info nce loss
        logits, labels = info_nce_loss(inner_featurep)
        i_loss = self.criterion(logits, labels)

        # 创建适合计算的pred
        bs = fp.shape[0] // 4
        nf = torch.zeros_like(fp)
        nf[:2*bs] = fp[:2*bs]
        nf[2*bs:3*bs] = fp[3*bs:]
        nf[3*bs:] = fp[2*bs:3*bs]

        # f loss
        logits, labels = info_nce_loss(nf)
        f_loss = self.criterion(logits, labels)

        return i_loss, f_loss


if __name__ == '__main__':

    set_determistic()  # 可复现性

    train_dirs = ["/home/xtanghao/THPycharm/dataset/diw5k",
                  "/home/xtanghao/THPycharm/dataset/Doc3D",
                  # "/home/xtanghao/THPycharm/dataset/Doc3DShade"
                  ]
    vgg_path = "./model_pretrained/vgg19.pkl"

    train_bs = 8
    valid_bs = 8
    num_workers = 8

    device = "cuda"

    lr = 1e-3
    factor = 0.8
    wd = 5e-4
    patience = 5
    min_lr = 1e-6
    epochs = 100

    img_shape = (448, 448)
    crop_transform = A.CropNonEmptyMaskIfExists(*img_shape)
    train_transform = A.Compose([  # 应用几何变换
        A.RandomBrightness(p=0.5),
        A.OneOf([
            A.ISONoise(),
            A.PixelDropout(),
            A.GaussianBlur()
        ], p=0.2),
        ToTensorV2(transpose_mask=True)
    ])

    train_dataset = ContrastiveDatasetv2(train_dirs, crop_transform=crop_transform, img_transform=train_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=train_bs, drop_last=False, shuffle=True,
                                  num_workers=num_workers)

    model = ContrasiveIllRec().to(device)
    # model.load_model("/home/xtanghao/DocRec/results/UnifyNet/03-19_00-30/model_24.pth")

    log_dir = f"./results/{model._get_name()}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}"
    writer = SummaryWriter(log_dir=log_dir)

    optimizer = AdamW(model.parameters(), lr=lr, amsgrad=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr,
                                               verbose=True)
    # ill_loss_f = IllLoss(vgg_path)
    loss_f = ContrasiveLoss()

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

    idx = 0
    for epoch in range(epochs):
        idx = train(idx, train_dataloader, model, writer, loss_f)
        # 保存模型
        torch.save(model.state_dict(), os.path.join(log_dir, "model_{}.pth").format(epoch))

        # 重置
        optimizer = AdamW(model.parameters(), lr=lr, amsgrad=True)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience,
                                                   min_lr=min_lr,
                                                   verbose=True)
        print(f"Epoch {epoch} finished.")

    print("train done. save model parameters done.")
    writer.close()
