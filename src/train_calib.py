import argparse
import os
from collections import OrderedDict
from datetime import date
from glob import glob

import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np

import archs
import losses
from dataset import Dataset
from metrics import iou_score
from unet_utils import AverageMeter, str2bool
from generate_subkaarts import generate_subkaarts
from utils.data_loading import SatteliteTrainDataset, SatteliteTestDataset, SatteliteValDataset
from ece_kde import get_ece_kde

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<512>"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run (how many sampling cycles)')
    parser.add_argument('--train_samples', default=1000, type=int, metavar='N',
                        help='number of total samples we take during one sampling cycle (epoch)')
    parser.add_argument('-b', '--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('-vb', '--val_batch_size', default=32, type=int,
                        metavar='N', help='validation-batch size (default: 50)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=15, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=256, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=256, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='CrossEntropyLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')

    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    parser.add_argument('--num_workers', default=1, type=int)
    
    #traindataset
    # parser.add_argument("-k", "--kaartbladen", default=['16','20','21','22'], nargs="+", type=str)
    parser.add_argument("-k", "--kaartbladen", default=generate_subkaarts([str(r) for r in list(range(1,44))])[0], nargs="+", type=str)
    parser.add_argument("-y", "--years", default=['2022'], nargs="+", type=str)
    parser.add_argument("-m", "--months", default=['03'], nargs="+", type=str)
    parser.add_argument("-r", "--root-dir", default='downloads_230703',type=str, required=False)
    # parser.add_argument("-r", "--root-dir", default='downloads',type=str, required=False)
    parser.add_argument(
        "-ps",
        "--patch-size",
        default=256,
        type=int,
        help="Size of train patches.",
    )
    parser.add_argument(
        "-vt",
        "--valid-threshold",
        default=0.3,
        type=float,
        help="Ratio of non-clouded area required to not mask-out a patch.",
    )
    
    #valdataset
    # parser.add_argument("-vk", "--vkaartbladen", default=['23','24','25'], nargs="+", type=str)
    parser.add_argument("-vk", "--vkaartbladen", default=generate_subkaarts([str(r) for r in list(range(1,44))])[1], nargs="+", type=str)
    parser.add_argument("-vy", "--vyears", default=['2022'], nargs="+", type=str)
    parser.add_argument("-vm", "--vmonths", default=['03'], nargs="+", type=str)
    # parser.add_argument("-vr", "--vroot-dir", default='downloads',type=str, required=False)
    parser.add_argument("-vr", "--vroot-dir", default='downloads_230703',type=str, required=False)
    parser.add_argument(
        "-vps",
        "--vpatch-size",
        default=256,
        type=int,
        help="Size of val patches.",
    )
    parser.add_argument(
        "-vpo",
        "--vpatch-offset",
        default=128,
        type=int,
        help="Offset between val patches.",
    )
    parser.add_argument(
        "-vvt",
        "--vvalid-threshold",
        default=0.3,
        type=float,
        help="Ratio of non-clouded area required to not mask-out a patch.",
    )

    parser.add_argument('-calib', '--ifcalibration', default=False, type=str2bool, help='if apply calibration to model')

    config = parser.parse_args()

    return config


def check_args(args):
    kaartbladen = args['kaartbladen']
    years = args['years']
    months = args['months']
    root_dir = args['root_dir']

    valid_kaartbladen = True
    if not len(kaartbladen):
        valid_kaartbladen = False
    for kaartblad in kaartbladen:
        print(kaartblad)
        if kaartblad not in [str(item) for item in range(1, 43)]:
            valid_kaartbladen = False
    if not valid_kaartbladen:
        raise ValueError(f"The provided kaartbladen: {kaartbladen} argument is invalid")

    valid_years = True
    if not len(years):
        valid_years = False
    for year in years:
        if year not in [str(item) for item in range(2010, 2023)]:
            valid_years = False
    if not valid_years:
        raise ValueError(f"The provided years: {years} argument is invalid")

    valid_months = True
    if not len(months):
        valid_months = False
    for month in months:
        if month not in [f"{item:02}" for item in range(1, 13)]:
            valid_months = False
    if not valid_months:
        raise ValueError(
            f"The provided months: {months} argument is invalid, the months must be strings representing strings in 'xy' format. eg. '03'"
        )

    valid_root_dir = True
    if not os.path.exists(root_dir):
        valid_root_dir = False
    if not valid_root_dir:
        raise ValueError(
            f"The provided root directory: {valid_root_dir} argument is invalid"
        )

    return


def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target


def train(config, train_loader, model, criterion, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()
    model = model.to(device)
    print("train starts:")
    # since the training data is generated on the go,
    # sample as many times as we need in one epoch, the number of samples in one epoch is user defined
    # and the epoch numbers are separately defined as well
    pbar = tqdm(total=config['train_samples'], position=0, leave=True)
    counter=0
    for sample in train_loader:
        sat = sample['sat']
        gt = sample['gt']
        valid_mask = sample["valid_mask"]
        gt_masked = gt * valid_mask
        gt_masked = make_one_hot(gt_masked, config['num_classes'])
        
        input = sat.to(device)
        target = gt_masked.to(device)
        # compute output
        if config['deep_supervision']:
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

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
        # release some GPU space
        torch.cuda.empty_cache()
        # sample times
        if counter == config['train_samples']:
            break
        else:
            counter = counter+1
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device: {device}")

    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    # switch to evaluate mode
    model.eval()
    model = model.to(device)
    print("validation starts:")
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader), position=0, leave=True)
        val_batch = config["val_batch_size"]
        for sample in val_loader:
            sat = sample["sat"]
            sat = torch.squeeze(sat, 0)
            gt = sample["gt"]
            valid_mask = sample["valid_mask"]
            gt_masked = gt * valid_mask
            gt_masked = torch.squeeze(gt_masked, 0)
            
            # split all patch tensors into batch_size numbered chunks
            sat_batches = torch.split(sat, val_batch, dim=0)
            gt_batches = torch.split(gt_masked, val_batch, dim=0)
            
            for sat_this, gt_this in zip(sat_batches, gt_batches):
                gt_this = make_one_hot(gt_this, config['num_classes'])
                
                input = sat_this.to(device)
                target = gt_this.to(device)
    
                # compute output
                if config['deep_supervision']:
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
    
                avg_meters['loss'].update(loss.item(), input.size(0))
                avg_meters['iou'].update(iou, input.size(0))
    
                postfix = OrderedDict([
                    ('loss', avg_meters['loss'].avg),
                    ('iou', avg_meters['iou'].avg),
                ])
            #update after all batches have gone through the model
            pbar.set_postfix(postfix)
            pbar.update(1)
            torch.cuda.empty_cache()
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def main():
    config = vars(parse_args())

    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)

    # define loss function (criterion)
    if config['loss'] == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss()
    elif config['loss'] == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = losses.__dict__[config['loss']]()

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'])


    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=1, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    # Data loading code
    # img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    # train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

    # train_transform = Compose([
    #     transforms.RandomRotate90(),
    #     transforms.Flip(),
    #     OneOf([
    #         transforms.HueSaturationValue(),
    #         transforms.RandomBrightness(),
    #         transforms.RandomContrast(),
    #     ], p=1),
    #     transforms.Resize(config['input_h'], config['input_w']),
    #     transforms.Normalize(),
    # ])

    # val_transform = Compose([
    #     transforms.Resize(config['input_h'], config['input_w']),
    #     transforms.Normalize(),
    # ])

    # train_dataset = Dataset(
    #     img_ids=train_img_ids,
    #     img_dir=os.path.join('inputs', config['dataset'], 'images'),
    #     mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
    #     img_ext=config['img_ext'],
    #     mask_ext=config['mask_ext'],
    #     num_classes=config['num_classes'],
    #     transform=train_transform)
    # val_dataset = Dataset(
    #     img_ids=val_img_ids,
    #     img_dir=os.path.join('inputs', config['dataset'], 'images'),
    #     mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
    #     img_ext=config['img_ext'],
    #     mask_ext=config['mask_ext'],
    #     num_classes=config['num_classes'],
    #     transform=val_transform)

    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=config['batch_size'],
    #     shuffle=True,
    #     num_workers=config['num_workers'],
    #     drop_last=True)
    # val_loader = torch.utils.data.DataLoader(
    #     val_dataset,
    #     batch_size=config['batch_size'],
    #     shuffle=False,
    #     num_workers=config['num_workers'],
    #     drop_last=False)
    
    
    from utils.constants import norm_hi_median as norm_hi
    from utils.constants import norm_lo_median as norm_lo

    # check_args(config)

    # kaartbladen = [str(item) for item in range(1, 43)]
    kaartbladen = config['kaartbladen']
    # Bad ones: 35 (mismatch, missaligned gt and sat), 43 (missaligned gt and sat) - the other small ones are good
    # kaartbladen.remove("35")
    # kaartbladen.remove("39")
    # kaartbladen.remove('43')
    # years = ["2021", "2022"]
    years = config['years']
    # months = [f"{item:02}" for item in range(1, 13)]
    months = config['months']
    # root_dir = "../generated_data"
    root_dir = config['root_dir']
    patch_size = config['patch_size']
    valid_threshold = config['valid_threshold']

    norm_hi = np.array(norm_hi)
    norm_lo = np.array(norm_lo)

    train_dataset = SatteliteTrainDataset(
        root_dir,
        kaartbladen,
        years,
        months,
        patch_size=patch_size,
        valid_threshold=valid_threshold,
        norm_hi=norm_hi,
        norm_lo=norm_lo
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        shuffle=True,
        drop_last=True
        )
    sample = next(iter(train_dataloader))
    
    sat = sample["sat"]
    gt = sample["gt"]
    valid_mask = sample["valid_mask"]
    print(sat.shape, gt.shape, valid_mask.shape)
    
    
    ##################################
    # kaartbladen = [str(item) for item in range(1, 43)]
    vkaartbladen = config['vkaartbladen']
    # Bad ones: 35 (mismatch, missaligned gt and sat), 43 (missaligned gt and sat) - the other small ones are good
    # kaartbladen.remove("35")
    # kaartbladen.remove("39")
    # kaartbladen.remove('43')
    # years = ["2021", "2022"]
    vyears = config['vyears']
    # months = [f"{item:02}" for item in range(1, 13)]
    vmonths = config['vmonths']
    # root_dir = "../generated_data"
    vroot_dir = config['vroot_dir']
    vpatch_size = config['vpatch_size']
    vpatch_offset = config['vpatch_offset']
    vvalid_threshold = config['vvalid_threshold']

    norm_hi = np.array(norm_hi)
    norm_lo = np.array(norm_lo)

    val_dataset = SatteliteValDataset(
        vroot_dir,
        vkaartbladen,
        vyears,
        vmonths,
        patch_size=vpatch_size,
        patch_offset=vpatch_offset,
        valid_threshold=vvalid_threshold,
        norm_hi=norm_hi,
        norm_lo=norm_lo
        )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=1, 
        shuffle=True
        )
    vsample = next(iter(val_dataloader))
    vsat = vsample["sat"]
    vgt = vsample["gt"]
    vvalid_mask = vsample["valid_mask"]
    vsat_full = vsample["sat_full"]
    vgt_full = vsample["gt_full"]
    vvalid_mask_full = vsample["valid_mask_full"]
    print(vsat.shape,vgt.shape,vvalid_mask.shape)

    tile_index = 0
    # plt.imshow(sat[0, tile_index].permute(1, 2, 0))
    
    
    
    
    
    
    
    
    

    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])

    best_iou = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_dataloader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_dataloader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)

        trigger += 1

        if val_log['iou'] > best_iou:
            today = date.today()
            dtoday = today.strftime("%Y%m%d")
            torch.save(model.state_dict(), f'models/%s/model{dtoday}.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
