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

class EceMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.ece_values = []
        self.valids = []
        self.avg_ece = 0
        self.ece_all = 0

    def reset(self):
        self.ece_values = []
        self.valids = []
        self.avg_ece = 0
        self.ece_all = 0

    def update(self, val, valid):
        self.ece_values.append(val)
        self.valids.append(valid)

    def report(self):
        self.ece_all = 0
        for i in range(len(self.ece_values)):
            self.ece_all += self.ece_values[i] * self.valids[i]
        self.avg_ece = self.ece_all / (sum(self.valids))
        return self.avg_ece
