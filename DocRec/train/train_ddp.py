import torch
import cv2
import os
import matplotlib.pyplot as plt
import datetime
import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, DistributedSampler, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataprocess.UVDoc.data_UVDoc import UVDocDataset
from loss_and_metrics.losses import IllLoss, GeoLoss
from loss_and_metrics.metric import SSIM
from models.UnifyNet import *
from utils.common import *
from dataprocess.Doc3D import data_Doc3D, dataset_docunet_valid


def train(idx, dataloader, model, writer, ill_loss_f, geo_loss_f, total_loss_f, save_result=None, cuda=True):
    model.train()

    scaler = GradScaler()
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

        img, label, bm, ill_img = data
        if cuda:
            img, label, bm, ill_img = img.cuda(), label.cuda(), bm.cuda(), ill_img.cuda()

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda' if cuda else 'cpu', dtype=torch.float16):
            # set_requires_grad([model.extractor, model.ill_decoder], True)
            # 前向传播
            ill_out, geo_output = model(img)
            # 结果矫正
            # label_only_geo = F.grid_sample(img, bm.permute(0, 2, 3, 1), align_corners=False)

            # X = torch.stack(torch.meshgrid(
            #     [torch.linspace(-1.0, 1.0, geo_output[0].shape[2], device=img.device),
            #      torch.linspace(-1.0, 1.0, geo_output[0].shape[3], device=img.device)], indexing='ij')[::-1],
            #                 dim=-1).contiguous().view(-1, 2).repeat(3, 1, 1)
            # tpsb = TPS(size=(img.shape[2], img.shape[3]), device=img.device)
            # # 变形处理
            # warped_grid_b = tpsb(X, geo_output[0].flatten(2).transpose(1, 2))
            # pred_only_geo = F.grid_sample(img, warped_grid_b, align_corners=False)
            # pred = F.grid_sample(ill_out, warped_grid_b, align_corners=False)
            # pred_only_geo = F.grid_sample(img, geo_output.permute(0, 2, 3, 1), align_corners=False)
            pred = F.grid_sample(ill_out, geo_output.permute(0, 2, 3, 1), align_corners=False)

            # 2 部分
            loss1 = ill_loss_f(ill_out, label)
            loss2 = geo_loss_f(geo_output, bm)
            loss3 = total_loss_f(pred, ill_img)
            # loss4 = total_loss_f(pred_only_geo, label_only_geo)
            # loss = loss2 + loss3 + loss4
            loss = loss1 * 0.8 + loss2 + loss3 * 0.2
            # loss = loss1 + loss2

        # 反向传播
        scaler.scale(loss).backward()
        # 更新模型参数
        scaler.step(optimizer)
        scaler.update()

        # 打印损失
        if idx % 10 == 0:
            writer.add_scalar("Train/loss1", loss1.item(), idx)
            writer.add_scalar("Train/loss2", loss2.item(), idx)
            writer.add_scalar("Train/loss3", loss3.item(), idx)
            # writer.add_scalar("Train/loss4", loss4.item(), idx)
            writer.add_scalar("Train/loss", loss.item(), idx)
            writer.add_scalar("Train/lr", optimizer.state_dict()['param_groups'][0]['lr'], idx)

            scheduler.step(loss)

        # 更新步骤计数器
        idx += 1

    if save_result != None:
        save_result(img.float(), ill_img.float(), pred.float())

    return idx


def evaluate(dataloader, model, valid_metric_f, save_result=None, cuda=True):
    model.eval()
    # 释放显存
    torch.cuda.empty_cache()
    metrics = []
    l1_losses = []
    for i, data in enumerate(dataloader):
        img, label = data
        if cuda:
            img, label = img.cuda(), label.cuda()
        # 前向传播
        with torch.no_grad():
            pred = model(img, return_final_result=True)
        # 计算loss
        metric = valid_metric_f(pred, label)
        l1_loss = F.l1_loss(pred, label)
        # 记录损失
        metrics.append(metric.item())
        l1_losses.append(l1_loss.item())

    if save_result != None:
        save_result(img, label, pred)

    return np.mean(metrics), np.mean(l1_losses)


if __name__ == '__main__':

    set_determistic()  # 可复现性
    local_rank = int(os.environ["LOCAL_RANK"])  # 也可以通过设置args.local_rank得到（见下文）
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl")

    root_dir = "../dataset/Doc3D"
    extra_dir = "../dataset/UVDoc_final"
    vgg_path = "./model_pretrained/vgg19.pkl"

    train_bs = 6
    valid_bs = 6
    num_workers = 6

    device = "cuda"

    lr = 2e-3
    factor = 0.8
    wd = 1e-4
    patience = 5
    min_lr = 1e-6
    epochs = 30

    img_shape = (448, 448)
    train_transform = A.Compose([  # 应用几何变换
        A.Resize(*img_shape),
        A.OneOf([
            A.GaussianBlur(),
            A.GaussNoise(),
            A.ISONoise(),
        ], p=0.1),
        ToTensorV2(transpose_mask=True)
    ])
    valid_transform = A.Compose([
        A.Resize(*img_shape),
        ToTensorV2(transpose_mask=True)
    ], is_check_shapes=False)

    train_dataset = data_Doc3D.Doc3D_All_Dataset(root_dir=root_dir, mode="train",
                                                 img_path_name="rembg", img_transform=train_transform)
    train_dataset_extra = UVDocDataset(extra_dir)
    train_dataset = ConcatDataset([train_dataset, train_dataset_extra])

    valid_dataset = dataset_docunet_valid.Valid_Dataset(img_transform=valid_transform)

    train_dataloader = DataLoader(train_dataset, batch_size=train_bs, drop_last=True,
                                  sampler=DistributedSampler(train_dataset, shuffle=True),
                                  num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_bs, drop_last=True,
                                  sampler=DistributedSampler(valid_dataset, shuffle=True),
                                  num_workers=num_workers)

    model = UnifyNet().to(device)
    model.load_model("/home/archlab/THPycharm/DocRec/results/UnifyNet/concat_8epochs/model_1.pth", not_use_parrel_trained=False)

    log_dir = f"./results/{model._get_name()}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}"
    writer = SummaryWriter(log_dir=log_dir)

    optimizer = AdamW([{'params': model.geo_decoder.parameters(), "lr": 5e-4},
                       {'params': model.ill_decoder.parameters(), "lr": 1e-3},
                       {'params': model.extractor.parameters()}], lr=lr, amsgrad=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr,
                                               verbose=True)
    # geo_loss_f = nn.L1Loss()
    geo_loss_f = GeoLoss(w_bnd=0.1, w_local=0.2)
    ill_loss_f = IllLoss(vgg_path)
    total_loss_f = IllLoss(vgg_path)
    valid_metric_f = SSIM()

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank,
                                                      find_unused_parameters=True)

    # 记录超参数信息
    writer.add_hparams({
        "train_bs": train_bs, "valid_bs": valid_bs, "num_workers": num_workers,
        "lr": lr, "factor": factor, "wd": wd, "patience": patience, "min_lr": min_lr, "epochs": epochs,
        "img_shape": str(img_shape), "train_transform": str(train_transform), "valid_transform": str(valid_transform),
        "model": str(model), "optimizer": str(optimizer), "scheduler": str(scheduler),
        "geo_loss_f": str(geo_loss_f), "ill_loss_f": str(ill_loss_f), "total_loss_f": str(total_loss_f),
        "valid_metric_f": str(valid_metric_f)
    }, {}, run_name="experiment setting")

    idx = 0
    for epoch in range(epochs):

        train_dataloader.sampler.set_epoch(epoch)
        valid_dataloader.sampler.set_epoch(epoch)

        idx = train(idx, train_dataloader, model, writer, ill_loss_f, geo_loss_f, total_loss_f,
                    save_result=lambda x, y, z: saveOutResult(x, y, z, f"train/{epoch}", writer))
        # 保存模型
        torch.save(model.state_dict(), os.path.join(log_dir, "model_{}.pth").format(epoch % 5))
        # 验证
        valid_ssim, valid_loss = evaluate(valid_dataloader, model, valid_metric_f,
                              save_result=lambda x, y, z: saveOutResult(x, y, z, f"valid/{epoch}", writer))

        # 重置
        if (epoch + 1) % 4 == 0:
            optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd, amsgrad=True)
            scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience,
                                                       min_lr=min_lr, verbose=True)
        writer.add_scalar("Valid/SSIM", valid_ssim, epoch)
        writer.add_scalar("Valid/L1Loss", valid_loss, epoch)
        print(f"Epoch {epoch} finished.")

    print("train done. save model parameters done.")
    writer.close()