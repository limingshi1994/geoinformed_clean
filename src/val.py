import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import yaml
from utils import losses
from tqdm import tqdm
from utils.eval_dataset import SatteliteEvalDataset
import matplotlib.pyplot as plt

import archs
from utils.metrics import iou_score
from unet_utils import AverageMeter
from train import make_one_hot
from matplotlib import cm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--name", default="dsb2018_96_NestedUNet_woDS", help="model name"
    )
    # parser.add_argument('--tkaartbladen', default=generate_subkaarts([str(r) for r in list(range(1,44))])[2],
    #                     help='Index of test kaartbladen')
    parser.add_argument(
        "-tk", "--tkaartbladen", default=list(range(1, 44)), nargs="+", type=str
    )
    parser.add_argument("-ty", "--tyears", default=["2022"], nargs="+", type=str)
    parser.add_argument("-tm", "--tmonths", default=["03"], nargs="+", type=str)
    parser.add_argument(
        "-tr", "--troot-dir", default="../downloads_230703", type=str, required=False
    )
    parser.add_argument(
        "-o", "--output-dir", default="../outputs", type=str, required=False
    )
    parser.add_argument(
        "-tps",
        "--tpatch-size",
        default=256,
        type=int,
        help="Size of test patches.",
    )
    parser.add_argument(
        "-plsf",
        "--preload_sat_flag",
        default=True,
        action="store_true",
        help="whether to preload satellite images",
    )
    parser.add_argument(
        "-plgf",
        "--preload_gt_flag",
        default=True,
        action="store_true",
        help="whether to preload ground truth",
    )
    parser.add_argument(
        "-plcf",
        "--preload_cloud_flag",
        default=True,
        action="store_true",
        help="whether to preload cloud masks",
    )

    args = parser.parse_args()

    return args


def validate(config, val_loader, model, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    avg_meters = {
        "loss": AverageMeterBatched(),
        "iou": AverageMeterBatched(),
        "acc": AverageSumsMeterBatched(),
    }

    # switch to evaluate mode
    model.eval()
    model = model.to(device)
    print("validation starts:")
    with torch.no_grad():
        pbar = tqdm(total=config["val_batches"], position=0, leave=True)
        counter = 0
        for sample in val_loader:
            sat = sample["sat"]
            gt = sample["gt"]
            valid_mask = sample["valid_mask"]
            cloud_mask = sample["cloud_mask"]
            label_mask = sample["label_mask"]

            # vvv HACK vvv
            # Works because we apply valid_mask on both the loss and the metric
            # gt - subtact one from all valid pixel labels
            gt[gt > 0] = gt[gt > 0] - 1
            # ^^^ HACK ^^^

            gt = make_one_hot(gt, config["num_classes"])

            input = sat.to(device)
            target = gt.to(device)
            valid_mask = valid_mask.to(device)

            if config["deep_supervision"]:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss, loss_track = criterion(output, target, mask=valid_mask)
                    loss += loss
                    loss_track += loss_track
                loss /= len(outputs)
                loss_track /= len(outputs)
                iou = iou_score(outputs[-1], target, mask=valid_mask)
                # acc = pixel_accuracy(outputs[-1], target, mask=valid_mask)
                correct, valid = pixel_accuracy(outputs[-1], target, mask=valid_mask)
            else:
                output = model(input)
                loss, loss_track = criterion(output, target, mask=valid_mask)
                iou = iou_score(output, target, mask=valid_mask)  # shape: bs
                # acc = pixel_accuracy(output, target, mask=valid_mask)  # shape: bs
                correct, valid = pixel_accuracy(output, target, mask=valid_mask)

            avg_meters["loss"].update(list(loss_track))
            avg_meters["iou"].update(list(iou))
            avg_meters["acc"].update(list(correct), list(valid))

            postfix = OrderedDict(
                [
                    ("loss", avg_meters["loss"].report()),
                    ("iou", avg_meters["iou"].report()),
                    ("acc", avg_meters["acc"].report()),
                ]
            )

            # update after all batches have gone through the model
            pbar.set_postfix(postfix)
            pbar.update(1)
            # release some GPU space
            torch.cuda.empty_cache()
            # sample times
            if counter == config["val_batches"]:
                break
            else:
                counter = counter + 1
        pbar.close()

    return OrderedDict(
        [
            ("loss", avg_meters["loss"].report()),
            ("iou", avg_meters["iou"].report()),
            ("acc", avg_meters["acc"].report()),
        ]
    )


def main():
    from utils.constants import norm_hi_median as norm_hi
    from utils.constants import norm_lo_median as norm_lo

    args = vars(parse_args())

    with open("models/%s/config.yml" % args["name"], "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print("-" * 20)
    for key in config.keys():
        print("%s: %s" % (key, str(config[key])))
    print("-" * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config["arch"])
    model = archs.__dict__[config["arch"]](
        config["num_classes"], config["input_channels"], config["deep_supervision"]
    )

    model = model.cuda()

    # Data loading code
    # img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # _, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load("models/%s/model230630.pth" % config["name"]))
    model.eval()

    # test_transform = Compose([
    #     transforms.Resize(config['input_h'], config['input_w']),
    #     transforms.Normalize(),
    # ])

    # val_dataset = Dataset(
    #     img_ids=val_img_ids,
    #     img_dir=os.path.join('inputs', config['dataset'], 'images'),
    #     mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
    #     img_ext=config['img_ext'],
    #     mask_ext=config['mask_ext'],
    #     num_classes=config['num_classes'],
    #     transform=val_transform)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=config['batch_size'],
    #     shuffle=False,
    #     num_workers=config['num_workers'],
    #     drop_last=False)

    tkaartbladen = args["tkaartbladen"]
    # Bad ones: 35 (mismatch, missaligned gt and sat), 43 (missaligned gt and sat) - the other small ones are good
    # kaartbladen.remove("35")
    # kaartbladen.remove("39")
    # kaartbladen.remove('43')
    # years = ["2021", "2022"]
    tyears = args["tyears"]
    # months = [f"{item:02}" for item in range(1, 13)]
    tmonths = args["tmonths"]
    # root_dir = "../generated_data"
    troot_dir = args["troot_dir"]
    tpatch_size = args["tpatch_size"]

    norm_hi = np.array(norm_hi)
    norm_lo = np.array(norm_lo)

    test_dataset = SatteliteEvalDataset(
        troot_dir,
        tkaartbladen,
        tyears,
        tmonths,
        patch_size=tpatch_size,
        norm_hi=norm_hi,
        norm_lo=norm_lo,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True
    )

    avg_meter = AverageMeter()

    if config["loss"] == "BCEWithLogitsLoss":
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config["loss"]]().cuda()

    for c in range(config["num_classes"]):
        os.makedirs(os.path.join("outputs", config["name"], str(c)), exist_ok=True)
    # with torch.no_grad():
    #     for sample in tqdm(test_dataloader, total=len(test_dataloader)):
    #         input = input.cuda()
    #         target = target.cuda()

    #         # compute output
    #         if config['deep_supervision']:
    #             output = model(input)[-1]
    #         else:
    #             output = model(input)

    colours = cm.get_cmap("viridis", config["num_classes"])
    cmap = colours(np.linspace(0, 1, config["num_classes"]))
    cmap[0, -1] = 0  # Set alpha for label 0 to be 0
    cmap[1:, -1] = 0.3  # Set the other alphas for the labels to be 0.3

    with torch.no_grad():
        pbar = tqdm(total=len(test_dataloader), position=0, leave=True)
        test_batch = 1
        outputid = 0
        for sample in test_dataloader:
            odir = f"output_{outputid}"
            if not os.path.exists(odir):
                os.makedirs(odir)

            sat = sample["sat"]
            sat = torch.squeeze(sat, 0)
            gt = sample["gt"]
            valid_mask = sample["valid_mask"]
            gt_masked = gt * valid_mask
            gt_masked = torch.squeeze(gt_masked, 0)

            # split all patch tensors into batch_size numbered chunks
            sat_batches = torch.split(sat, test_batch, dim=0)
            gt_batches = torch.split(gt_masked, test_batch, dim=0)

            imid = 0

            for sat_this, gt_this in zip(sat_batches, gt_batches):
                gt_this = make_one_hot(gt_this, config["num_classes"])

                input = sat_this.cuda()
                target = gt_this.cuda()

                # compute output
                if config["deep_supervision"]:
                    outputs = model(input)
                    loss = 0
                    for output in outputs:
                        loss += criterion(output, target)
                    loss /= len(outputs)
                    iou = iou_score(outputs[-1], target)
                else:
                    output = model(input)
                    loss = criterion(output, target)
                    iou = iou_score(output, target)

                avg_meter.update(iou, input.size(0))

                output = torch.softmax(output, dim=1)

                pred = torch.argmax(output, dim=1).cpu().numpy()

                # visualization
                currentfig = plt.figure(dpi=1200)
                satim = sat_this.cpu().numpy()
                satim = np.squeeze(satim, axis=0)
                satim = np.moveaxis(satim, 0, -1)

                pred = np.squeeze(pred, axis=0)
                ovlypred = cmap[pred.flatten()]
                Ra, Ca = pred.shape[:2]
                ovlypred = ovlypred.reshape((Ra, Ca, -1))

                satfig = plt.figure(dpi=1200)
                satfig = plt.imshow(satim)
                plt.savefig(f"{odir}/satimg_{imid}.png", dpi=1200)
                # plt.show(satfig)

                currentfig = plt.imshow(satim)
                currentfig = plt.imshow(ovlypred)
                plt.savefig(f"{odir}/merged_{imid}.png", dpi=1200)
                # plt.show(currentfig)
                imid = imid + 1

            outputid = outputid + 1

    print("IoU: %.4f" % avg_meter.avg)

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
