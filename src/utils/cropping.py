import math
import random

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


def random_crop_old(
    img,
    gt,
    valid_mask,
    cloud_mask, 
    label_mask,
    height,
    width,
    padding_value_img=1,
    padding_value_gt=0,
    padding_value_valid_mask=0,
    padding_value_cloud_mask=1,
    padding_value_label_mask=0,
):


    assert (
        type(img) == torch.Tensor
        and type(gt) == torch.Tensor
        and type(valid_mask) == torch.Tensor
        and type(cloud_mask) == torch.Tensor
        and type(label_mask) == torch.Tensor
    ), "Must have torch.Tensor as inputs"
    assert (
        len(img.shape) == 3 and len(gt.shape) == 3 and len(valid_mask.shape) and len(cloud_mask.shape) and len(label_mask.shape)
    ), "Must have three dimensional image and masks"
    assert (
        img.shape[1] == gt.shape[1] == valid_mask.shape[1] == cloud_mask.shape[1] == label_mask.shape[1]
    ), "The image and masks must have the same height"
    assert (
        img.shape[2] == gt.shape[2] == valid_mask.shape[2] == cloud_mask.shape[2] == label_mask.shape[2]
    ), "The image and masks must have the same width"

    if height > img.shape[1]:
        y = 0
        pad_height = math.ceil((height - img.shape[1]) / 2)
        padding = (0, 0, pad_height, pad_height)

        img = F.pad(img, padding, "constant", padding_value_img)
        gt = F.pad(gt, padding, "constant", padding_value_gt)
        valid_mask = F.pad(valid_mask, padding, "constant", padding_value_valid_mask)
        cloud_mask = F.pad(cloud_mask, padding, "constant", padding_value_cloud_mask)
        label_mask = F.pad(label_mask, padding, "constant", padding_value_label_mask)

    if width > img.shape[2]:
        x = 0
        pad_width = math.ceil((width - img.shape[2]) / 2)
        padding = (pad_width, pad_width, 0, 0)

        img = F.pad(img, padding, "constant", padding_value_img)
        gt = F.pad(gt, padding, "constant", padding_value_gt)
        valid_mask = F.pad(valid_mask, padding, "constant", padding_value_valid_mask)
        cloud_mask = F.pad(cloud_mask, padding, "constant", padding_value_cloud_mask)
        label_mask = F.pad(label_mask, padding, "constant", padding_value_label_mask)


    y = random.randint(0, img.shape[1] - height)
    x = random.randint(0, img.shape[2] - width)

    img = img[:, y : y + height, x : x + width]
    gt = gt[:, y : y + height, x : x + width]
    valid_mask = valid_mask[:, y : y + height, x : x + width]
    cloud_mask = cloud_mask[:, y : y + height, x : x + width]
    label_mask = label_mask[:, y : y + height, x : x + width]
    return img, gt, valid_mask, cloud_mask, label_mask


def random_crop(
    img,
    gt,
    valid_mask,
    cloud_mask,
    label_mask,
    height,
    width
):
    assert (
        type(img) == torch.Tensor
        and type(gt) == torch.Tensor
        and type(valid_mask) == torch.Tensor
        and type(cloud_mask) == torch.Tensor
        and type(label_mask) == torch.Tensor
    ), "Must have torch.Tensor as inputs"
    assert (
        len(img.shape) == 3 and len(gt.shape) == 3 and len(valid_mask.shape) and len(cloud_mask.shape) and len(label_mask.shape)
    ), "Must have three dimensional image and masks"
    assert (
        img.shape[1] == gt.shape[1] == valid_mask.shape[1] == cloud_mask.shape[1] == label_mask.shape[1]
    ), "The image and masks must have the same height"
    assert (
        img.shape[2] == gt.shape[2] == valid_mask.shape[2] == cloud_mask.shape[2] == label_mask.shape[2]
    ), "The image and masks must have the same width"

    h, w = img.shape[1], img.shape[2]

    if height > h:
        pad_height_top = math.ceil((height - h) / 2)
        pad_height_bottom = math.floor((height - h) / 2)
    else:
        pad_height_top = 0
        pad_height_bottom = 0

    if width > w:
        pad_width_left = math.ceil((width - w) / 2)
        pad_width_right = math.floor((width - w) / 2)
    else:
        pad_width_left = 0
        pad_width_right = 0

    padding = (pad_width_left, pad_width_right, pad_height_top, pad_height_bottom)
    if any(padding):
        img = F.pad(img, padding, "reflect")
        gt = F.pad(gt, padding, "reflect")
        valid_mask = F.pad(valid_mask, padding, "reflect")
        cloud_mask = F.pad(cloud_mask, padding, "reflect")
        label_mask = F.pad(label_mask, padding, "reflect")

    if height > h:
        y = 0
    else:
        y = random.randint(0, h - height)

    if width > w:
        x = 0
    else:
        x = random.randint(0, w - width)

    img = img[:, y : y + height, x : x + width]
    gt = gt[:, y : y + height, x : x + width]
    valid_mask = valid_mask[:, y : y + height, x : x + width]
    cloud_mask = cloud_mask[:, y : y + height, x : x + width]
    label_mask = label_mask[:, y : y + height, x : x + width]
    return img, gt, valid_mask, cloud_mask, label_mask