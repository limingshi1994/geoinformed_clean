import os
import argparse
import numpy as np

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from utils.data_loading import SatteliteTrainDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", "--kaartbladen", default=['16'], nargs="+", type=str)
    parser.add_argument("-y", "--years", default=['2022'], nargs="+", type=str)
    parser.add_argument("-m", "--months", default=['03'], nargs="+", type=str)
    parser.add_argument("-r", "--root-dir", default='../downloads_230703',type=str)
    parser.add_argument(
        "-ps",
        "--patch-size",
        default=256,
        type=int,
        help="Size of patches.",
    )
    parser.add_argument(
        "-vt",
        "--valid-threshold",
        default=0.3,
        type=float,
        help="Ratio of non-clouded area required to not mask-out a patch.",
    )
    args = parser.parse_args()
    return args


def check_args(args):
    kaartbladen = args.kaartbladen
    years = args.years
    months = args.months
    root_dir = args.root_dir

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


def main():
    from utils.constants import norm_hi_median as norm_hi
    from utils.constants import norm_lo_median as norm_lo

    args = get_args()
    check_args(args)

    # kaartbladen = [str(item) for item in range(1, 43)]
    kaartbladen = args.kaartbladen
    # Bad ones: 35 (mismatch, missaligned gt and sat), 43 (missaligned gt and sat) - the other small ones are good
    # kaartbladen.remove("35")
    # kaartbladen.remove("39")
    # kaartbladen.remove('43')
    # years = ["2021", "2022"]
    years = args.years
    # months = [f"{item:02}" for item in range(1, 13)]
    months = args.months
    # root_dir = "../generated_data"
    root_dir = args.root_dir
    patch_size = args.patch_size
    valid_threshold = args.valid_threshold

    norm_hi = np.array(norm_hi)
    norm_lo = np.array(norm_lo)

    dataset = SatteliteTrainDataset(
        root_dir,
        kaartbladen,
        years,
        months,
        patch_size=patch_size,
        valid_threshold=valid_threshold,
        norm_hi=norm_hi,
        norm_lo=norm_lo,
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    sample = next(iter(dataloader))
    
    print(len(dataloader))
    print(sample['sat'].shape)
    print(type(sample['sat']))
    print(sample['gt'].shape)
    print(type(sample['gt']))
    
    sat = sample["sat"]
    gt = sample["gt"]
    valid_mask = sample["valid_mask"]
    print(sat.shape, gt.shape, valid_mask.shape)

    plt.imshow(sat[0].permute(1, 2, 0))
    gt_masked = gt.permute(0,2,3,1) * valid_mask.permute(0,2,3,1)
    plt.imshow(
        gt_masked[0],
        alpha=0.3,
    )
    plt.title("Sattelite Patch with Segmentation Mask")
    plt.show()


if __name__ == "__main__":
    main()
