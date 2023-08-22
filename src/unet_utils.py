import argparse


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class AverageMeterBatched(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.values = []

    def reset(self):
        self.values = []

    def update(self, val):
        self.values.extend(val)

    def report(self):
        return sum(self.values) / len(self.values)

class AverageSumsMeterBatched(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.values = []
        self.valids = []

    def reset(self):
        self.values = []
        self.valids = []

    def update(self, val, valid):
        self.values.extend(val)
        self.valids.extend(valid)

    def report(self):
        return sum(self.values) / sum(self.valids)
