import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target, **kwargs):

    smooth = 1e-5
    bs = output.shape[0]

    if "mask" in kwargs:
        mask = kwargs["mask"]
    else:
        mask = None

    output = torch.softmax(output, dim=1)
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = output_ & target_
    if mask is not None:
        intersection = intersection * mask
    intersection = intersection.view(bs, -1).sum(dim=1).detach().cpu().numpy()

    union = output_ | target_
    if mask is not None:
        union = union * mask
    union = union.view(bs, -1).sum(dim=1).detach().cpu().numpy()

    return (intersection + smooth) / (union + smooth)


def dice_coef(output, target):
    smooth = 1e-5
    bs = output.shape[0]

    output = torch.sigmoid(output).view(-1).data.cpu()
    target = target.view(-1).data.cpu()
    intersection = (output * target).view(bs, -1).sum(dim=1).numpy()
    output = output.numpy()
    target = target.numpy()

    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def pixel_accuracy(output, target, **kwargs):

    if "mask" in kwargs:
        mask = kwargs["mask"]
    else:
        mask = None

    bs = output.shape[0]
    output = torch.softmax(output, dim=1)
    predicted = torch.argmax(output, dim=1)
    label = torch.argmax(target, dim=1)

    if mask is not None:
        # Comparing predicted and label only where mask_tensor is True
        correct_where_valid = (predicted == label) & (mask.squeeze(1))
        correct = correct_where_valid.float()
    else:
        correct = (predicted == label).float()
        correct = correct * (mask.squeeze(1))

    # Exclude the invalid pixels from calculation of correctness : correct_where_valid/valid_pixels
    if mask is not None:
        correct = correct.sum().item() / mask.float().sum().item()
    else:
        correct = correct.view(bs, -1).mean(dim=1).detach().cpu().numpy()

    return correct
