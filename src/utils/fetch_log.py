import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import segmentation_models_pytorch as smp


def fetch_log(arch, encoder, pretrained, batch_number, batch_size):
    model_dir = "/esat/gebo/mli1/pycharmproj/geoinformed_clean/outputs/models/dsb2018_96_NestedUNet_woDS/"
    train_name = f"arch_{arch}_enc_{encoder}_train_{batch_number}x{batch_size}_val_{batch_number}x{batch_size}"
    log_dir = model_dir + train_name

    csvs = glob.glob(log_dir + "/*.csv")
    csvs.sort(key=os.path.getmtime, reverse=True)
    csv = csvs[0]

    ymls = glob.glob(log_dir + "/*.yml")
    ymls.sort(key=os.path.getmtime, reverse=True)
    yml = ymls[0]

    chkpts = glob.glob(log_dir + "/*.pth")
    chkpts.sort(key=os.path.getmtime, reverse=True)
    chkpt = chkpts[0]

    # load model
    architecture = getattr(smp, arch)
    model = architecture(
        encoder_name=encoder,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights=pretrained,  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=14,  # model output channels (number of classes in your dataset)
    )
    model.load_state_dict(torch.load(chkpt))
    model.eval()

    # get parameters count
    params = sum(p.numel() for p in model.parameters())
    print(f'parameters count of this model is {params}')

    # get loss and accuracy data
    df = pd.read_csv(
        f"{csv}", usecols=["epoch", "val_loss", "val_acc"]
    )
    epoch = df["epoch"]
    loss = df["val_loss"]
    acc = df["val_acc"]
    ema_loss = loss.ewm(com=30).mean()
    ema_acc = acc.ewm(com=30).mean()

    # plot
    # loss_curve = plt.plot(np.log(epoch), np.log(ema_loss), label=f"{arch}_{encoder}_loss_log")
    loss_curve = plt.plot(epoch, ema_loss, label=f"{arch}_{encoder}_loss")
    acc_curve = plt.plot(epoch, ema_acc, label=f"{arch}_{encoder}_acc")
    legended = plt.legend()

    return yml, loss_curve, acc_curve, legended
