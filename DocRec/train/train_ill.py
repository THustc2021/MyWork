import datetime
import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from torch.cuda.amp import GradScaler
from torch.optim import lr_scheduler
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss_and_metrics.losses import IllLoss
from loss_and_metrics.metric import SSIM
from models.IllRec.ill_decoderv1 import *
from utils.common import *
from dataprocess.Doc3D import data_Doc3D, dataset_docunet_valid


def train(idx, dataloader, model, writer, ill_loss_f, save_result=None, cuda=True):
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
            ill_out = model(img)
            # 光照损失
            loss = ill_loss_f(ill_out, label)

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
        save_result(img.float(), label.float(), ill_out.float())

    return idx


def evaluate(dataloader, model, valid_loss_f, save_result=None, cuda=True):
    model.eval()
    # 释放显存
    torch.cuda.empty_cache()
    losses = []
    for i, data in enumerate(dataloader):
        img, label = data
        if cuda:
            img, label = img.cuda(), label.cuda()
        # 前向传播
        with torch.no_grad():
            pred = model(img)
        # 计算loss
        loss = valid_loss_f(pred, label)
        # 记录损失
        losses.append(loss.item())

    if save_result != None:
        save_result(img, label, pred)

    return np.mean(losses)


if __name__ == '__main__':

    set_determistic()  # 可复现性

    root_dir = "/home/xtanghao/THPycharm/dataset/Doc3D"
    extra_dir = "/home/xtanghao/THPycharm/dataset/Doc3DShade"
    vgg_path = "../model_pretrained/vgg19.pkl"

    train_bs = 48
    valid_bs = 48
    num_workers = 8

    device = "cuda"

    lr = 5e-3
    factor = 0.8
    wd = 5e-4
    patience = 5
    min_lr = 1e-6
    epochs = 30

    img_shape = (224, 224)
    train_transform = A.Compose([  # 应用几何变换
        A.Resize(*img_shape),
        A.GaussianBlur(p=0.1),
        ToTensorV2(transpose_mask=True)
    ])
    valid_transform = A.Compose([
        A.Resize(*img_shape),
        ToTensorV2(transpose_mask=True)
    ], is_check_shapes=False)

    train_dataset = data_Doc3D.DocIll_dataset(root_dir=root_dir, mode="train", img_path_name="rembg",
                                                 label_path_name="alb", img_transform=train_transform)
    valid_dataset = data_Doc3D.DocIll_dataset(root_dir=root_dir, mode="valid", img_path_name="rembg",
                                                 label_path_name="alb", img_transform=valid_transform)
    train_dataset_extra = data_Doc3D.DocIll_dataset(root_dir=extra_dir, mode="train", img_path_name="img",
                                                    label_path_name="alb", img_transform=train_transform)
    valid_dataset_extra = data_Doc3D.DocIll_dataset(root_dir=extra_dir, mode="valid", img_path_name="img",
                                                    label_path_name="alb", img_transform=valid_transform)
    train_dataset = ConcatDataset([train_dataset, train_dataset_extra])
    valid_dataset = ConcatDataset([valid_dataset, valid_dataset_extra])

    train_dataloader = DataLoader(train_dataset, batch_size=train_bs, drop_last=False, shuffle=True,
                                  num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_bs, drop_last=True, shuffle=True,
                                  num_workers=num_workers)

    model = IllRec().to(device)
    # model.load_model("/home/xtanghao/DocRec/results/UnifyNet/03-19_00-30/model_24.pth")

    log_dir = f"../results/{model._get_name()}/{datetime.datetime.now().strftime('%m-%d_%H-%M')}"
    writer = SummaryWriter(log_dir=log_dir)

    optimizer = AdamW(model.parameters(), lr=lr, amsgrad=True)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr,
                                               verbose=True)
    ill_loss_f = IllLoss(vgg_path)

    # 记录超参数信息
    writer.add_hparams({
        "train_bs": train_bs, "valid_bs": valid_bs, "num_workers": num_workers,
        "lr": lr, "factor": factor, "wd": wd, "patience": patience, "min_lr": min_lr, "epochs": epochs,
        "img_shape": str(img_shape), "train_transform": str(train_transform), "valid_transform": str(valid_transform),
        "model": str(model), "optimizer": str(optimizer), "scheduler": str(scheduler),
        "ill_loss_f": str(ill_loss_f)
    }, {}, run_name="experiment setting")
    # # 记录模型静态图
    # writer.add_graph(model, torch.rand((2, 3, 448, 448)).cuda(), verbose=True)
    # writer.flush()

    idx = 0
    for epoch in range(epochs):
        idx = train(idx, train_dataloader, model, writer, ill_loss_f,
                    save_result=lambda x, y, z: saveOutResult(x, y, z, f"train/{epoch}", writer))
        # 保存模型
        torch.save(model.state_dict(), os.path.join(log_dir, "model_{}.pth").format(epoch))
        # 验证
        valid_loss = evaluate(valid_dataloader, model, ill_loss_f,
                              save_result=lambda x, y, z: saveOutResult(x, y, z, f"valid/{epoch}", writer))

        # 重置
        optimizer = AdamW(model.parameters(), lr=lr, amsgrad=True)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience,
                                                   min_lr=min_lr,
                                                   verbose=True)
        writer.add_scalar("Valid/Loss", valid_loss, epoch)
        print(f"Epoch {epoch} finished.")

    print("train done. save model parameters done.")
    writer.close()
