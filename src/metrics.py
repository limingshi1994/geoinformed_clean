import numpy as np
import torch
import torch.nn.functional as F


def iou_score(output, target):
    smooth = 1e-5
    bs = output.shape[0]

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu()
    if torch.is_tensor(target):
        target = target.data.cpu()
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).view(bs, -1).sum(dim=1).numpy()
    union = (output_ | target_).view(bs, -1).sum(dim=1).numpy()

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

    if "label_mask" in kwargs:
        label_mask = kwargs["label_mask"]
    else:
        label_mask = None

    bs = output.shape[0]

    if torch.is_tensor(output):
        output = torch.softmax(output, dim=1).data.cpu()
    if torch.is_tensor(target):
        target = target.data.cpu()
    predicted = torch.argmax(output, dim=1)
    label = torch.argmax(target, dim=1)
    correct = (predicted == label).float()
    if label_mask is not None:
        correct = correct * label_mask.squeeze(1)
    correct = correct.view(bs, -1).mean(dim=1).numpy()
    return correct
