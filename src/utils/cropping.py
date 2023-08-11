import math
import random

import torch
import torchvision.transforms as transforms
import torch.nn.functional as F


def random_crop(
    img,
    mask,
    valid_mask,
    height,
    width,
    padding_value_img=1,
    padding_value_mask=0,
    padding_value_valid_mask=0,
):


    assert (
        type(img) == torch.Tensor
        and type(mask) == torch.Tensor
        and type(valid_mask) == torch.Tensor
    ), "Must have torch.Tensor as inputs"
    assert (
        len(img.shape) == 3 and len(mask.shape) == 3 and len(valid_mask.shape)
    ), "Must have three dimensional image and masks"
    assert (
        img.shape[1] == mask.shape[1] == valid_mask.shape[1]
    ), "The image and masks must have the same height"
    assert (
        img.shape[2] == mask.shape[2] == valid_mask.shape[2]
    ), "The image and masks must have the same width"

    if height > img.shape[1]:
        y = 0
        pad_height = math.ceil((height - img.shape[1]) / 2)
        padding = (0, 0, pad_height, pad_height)
        img = F.pad(img, padding, "constant", padding_value_img)
        mask = F.pad(mask, padding, "constant", padding_value_mask)
        valid_mask = F.pad(valid_mask, padding, "constant", padding_value_valid_mask)

    if width > img.shape[2]:
        x = 0
        pad_width = math.ceil((width - img.shape[2]) / 2)
        padding = (pad_width, pad_width, 0, 0)
        img = F.pad(img, padding, "constant", padding_value_img)
        mask = F.pad(mask, padding, "constant", padding_value_mask)
        valid_mask = F.pad(valid_mask, padding, "constant", padding_value_valid_mask)

    y = random.randint(0, img.shape[1] - height)
    x = random.randint(0, img.shape[2] - width)

    img = img[:, y : y + height, x : x + width]
    mask = mask[:, y : y + height, x : x + width]
    valid_mask = valid_mask[:, y : y + height, x : x + width]
    return img, mask, valid_mask
