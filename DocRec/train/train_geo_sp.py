import datetime
import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataprocess.Doc3D.data_tps import TPSDataset
from loss_and_metrics.losses import IllLoss, GeoLoss
from loss_and_metrics.metric import SSIM
from models.Contrasive.GeoRec_cptr import *
from utils.common import *
from dataprocess.Doc3D import data_Doc3D, dataset_docunet_valid


def train(idx, dataloader, model, writer, geo_loss_f, save_result=None, cuda=True):
    model.train()

    scaler = GradScaler()
    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):

        img, label = data
        if cuda:
            img, label = img.cuda(), label.cuda()

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda' if cuda else 'cpu', dtype=torch.float16):
            # set_requires_grad([model.extractor, model.ill_decoder], True)
            # 前向传播
            # 先训练光照部分
            pn = model(img)
            # 光照损失
            loss = geo_loss_f(pn, label)

        # 反向传播
        scaler.scale(loss).backward()
        # 更新模型参数
        scaler.step(optimizer)
        scaler.update()

        # 打印损失
        if idx % 10 == 0:
            writer.add_scalar("Train/loss", loss.item(), idx)
            writer.add_scalar("Train/lr", optimizer.state_dict()['param_groups'][0]['lr'], idx)

            scheduler.step(loss)

        # 更新步骤计数器
        idx += 1

    if save_result != None:
        try:
            geo_out = warptps(img[:3], train_dataset.points_ori_t_norm[None].repeat(3, 1, 1), pn[:3])
            label_out = warptps(img[:3], train_dataset.points_ori_t_norm[None].repeat(3, 1, 1), label[:3])
            save_result(img.float(), label_out.float(), geo_out.float())
        except:
            pass

    return idx


def evaluate(dataloader, model, valid_metric_f, save_result=None, cuda=True):
    model.eval()
    # 释放显存
    torch.cuda.empty_cache()
    metrics = []
    for i, data in enumerate(dataloader):
        img, label = data
        if cuda:
            img, label = img.cuda(), label.cuda()
        # 前向传播
        with torch.no_grad():
            pred = model(img)
            try:
                geo_out = warptps(img[:3], train_dataset.points_ori_t_norm[None].repeat(3, 1, 1), pred[:3])
                label_out = warptps(img[:3], train_dataset.points_ori_t_norm[None].repeat(3, 1, 1), label[:3])
            except:
                continue
        # 计算metric
        metric = valid_metric_f(geo_out, label_out)
        # 记录损失
        metrics.append(metric.item())

    if save_result != None:
        save_result(img, label_out, geo_out)

    return np.mean(metrics)


if __name__ == '__main__':

    set_determistic()  # 可复现性

    train_dirs = [
        # "/home/xtanghao/THPycharm/dataset/Doc3D/ill_gt",
        "/home/xtanghao/THPycharm/dataset/DocUnet Benchmark/scan",
        "/home/xtanghao/THPycharm/dataset/DIR300/gt",
        "/home/xtanghao/THPycharm/dataset/DocReal/scanned",
        "/home/xtanghao/THPycharm/dataset/WarpDoc/digital",
        "/home/xtanghao/THPycharm/dataset/WarpDoc/digital_margin",
        "/home/xtanghao/THPycharm/dataset/UVDoc_final/textures",
        "/home/xtanghao/THPycharm/dataset/docs-sm"
    ]
    vgg_path = "../model_pretrained/vgg19.pkl"

    train_bs = 64
    valid_bs = 64
    num_workers = 8
    pnpaxis = 8

    device = "cuda"

    lr = 1e-3
    factor = 0.8
    wd = 5e-4
    patience = 5
    min_lr = 1e-6
    epochs = 200

    img_shape = (448, 448)
    train_transform = A.Compose([  # 应用几何变换
        A.PadIfNeeded(*img_shape),
        A.RandomCrop(*img_shape),
        A.OneOf(
        [
            A.ColorJitter(),
            A.PixelDropout(),
            A.GaussianBlur(),
            A.ISONoise()
         ],
        p=0.1),
    ])
    valid_transform = A.Compose([
        A.Resize(*img_shape),
    ], is_check_shapes=False)

    train_dataset = TPSDataset(root_dirs=train_dirs, img_transform=train_transform, num_points_per_axis=pnpaxis)
    valid_dataset = TPSDataset(root_dirs=["/home/xtanghao/THPycharm/dataset/DocUnet Benchmark/scan",
                                          "/home/xtanghao/THPycharm/dataset/DIR300/gt"], img_transform=valid_transform,
                               num_points_per_axis=pnpaxis)

    train_dataloader = DataLoader(train_dataset, batch_size=train_bs, drop_last=False, shuffle=True,
                                  num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_bs, drop_last=False, shuffle=True,
                                  num_workers=num_workers)

    model = GeoRecCPTR(pnpaxis, img_shape).to(device)
    # model.load_model("/home/xtanghao/DocRec/results/UnifyNet/03-19_00-30/model_24.pth")

    log_dir = f"../results/{model._get_name()}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}"
    writer = SummaryWriter(log_dir=log_dir)

    optimizer = AdamW(model.parameters(), lr=lr, amsgrad=True)
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr,
    #                                            verbose=True)
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=100, eta_min=min_lr)
    loss_f = nn.L1Loss()
    valid_metric = SSIM()

    # 记录超参数信息
    writer.add_hparams({
        "train_bs": train_bs, "valid_bs": valid_bs, "num_workers": num_workers,
        "lr": lr, "factor": factor, "wd": wd, "patience": patience, "min_lr": min_lr, "epochs": epochs,
        "img_shape": str(img_shape), "train_transform": str(train_transform), "valid_transform": str(valid_transform),
        "model": str(model), "optimizer": str(optimizer), "scheduler": str(scheduler),
        "geo_loss_f": str(loss_f)
    }, {}, run_name="experiment setting")
    # # 记录模型静态图
    # writer.add_graph(model, torch.rand((2, 3, 224, 224)).cuda(), verbose=True)
    # writer.flush()

    idx = 0
    for epoch in range(epochs):
        idx = train(idx, train_dataloader, model, writer, loss_f,
                    save_result=lambda x, y, z: saveOutResult(x, y, z, f"train/{epoch}", writer))
        # 保存模型
        torch.save(model.state_dict(), os.path.join(log_dir, "model_{}.pth").format(epoch % 5))
        # 验证
        valid_metric = evaluate(valid_dataloader, model, loss_f,
                              save_result=lambda x, y, z: saveOutResult(x, y, z, f"valid/{epoch}", writer))

        # 重置
        optimizer = AdamW(model.parameters(), lr=lr, amsgrad=True)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience,
                                                   min_lr=min_lr,
                                                   verbose=True)
        writer.add_scalar("Valid/SSIM", valid_metric, epoch)
        print(f"Epoch {epoch} finished.")

    print("train done. save model parameters done.")
    writer.close()
