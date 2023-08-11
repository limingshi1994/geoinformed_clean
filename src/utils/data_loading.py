import os
import re
import copy
import random

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from utils.gio import load_tiff
from utils.cropping import random_crop
from utils.tiling import split_into_tiles
from utils.normalization import satellite_normalization_with_cloud_masking


class SatteliteTrainDataset(nn.Module):
    def __init__(
        self,
        root_dir,
        kaartbladen,
        years,
        months,
        patch_size=256,
        valid_threshold=0.5,
        norm_hi=None,
        norm_lo=None,
    ):
        """
        Arguments:
            root_dir (string): Directory with all the images.
                The structure of the root dir should be like:
                    root_dir/
                        data_gt\
                            gt_kaartblad_1.tiff
                            ...
                            gt_kaartblad_43.tiff
                        
                        data_sat\
                            kaartblad_1
                                kaartblad_1_202X-XX-XXZ.tif
                                ...
                                kaartblad_1_202X-XX-XXZ.tif
                            ...
                            kaartblad_43
                                kaartblad_43_202X-XX-XXZ.tif
                                ...
                                kaartblad_43_202X-XX-XXZ.tif
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.gt_dir = f"{root_dir}/data_gt"
        self.sat_dir = f"{root_dir}/data_sat"
        self.kaartbladen = kaartbladen
        self.kaartbladen_names = [f"kaartblad_{item}" for item in kaartbladen]
        self.years = years
        self.months = months

        self.patch_size = patch_size
        self.norm_hi = norm_hi
        self.norm_lo = norm_lo
        self.valid_threshold = valid_threshold

        self.data_dict = {}
        self.build_data_dict()
        self.filter_by_year(years)
        self.filter_by_month(months)

    def build_data_dict(self):
        print("Building the data dictionary...")
        for gt_file in os.listdir(self.gt_dir):
            gt_file_path = os.path.join(self.gt_dir, gt_file)
            kaartblad_name = re.findall(r"(kaartblad_\w+-\w).", gt_file)[0]
            if kaartblad_name in self.kaartbladen_names:
                self.data_dict[kaartblad_name] = {}
                self.data_dict[kaartblad_name]["gt_path"] = gt_file_path
                self.data_dict[kaartblad_name]["satellite_images"] = {}
                self.data_dict[kaartblad_name]["cloud_masks"] = {}
                for file in os.listdir(os.path.join(self.sat_dir, kaartblad_name)):
                    if file.endswith(".tif"):
                        sat_file_path = os.path.join(self.sat_dir, kaartblad_name, file)
                        year, month, day = re.findall(
                            r"(\d{4})-(\d{1,2})-(\d{1,2})Z", file
                        )[0]
                        if (
                            year
                            not in self.data_dict[kaartblad_name]["satellite_images"]
                        ):
                            self.data_dict[kaartblad_name]["satellite_images"][
                                year
                            ] = {}

                        if (
                            month
                            not in self.data_dict[kaartblad_name]["satellite_images"][
                                year
                            ]
                        ):
                            self.data_dict[kaartblad_name]["satellite_images"][year][
                                month
                            ] = {}

                        self.data_dict[kaartblad_name]["satellite_images"][year][month][
                            day
                        ] = sat_file_path
                    elif file.endswith(".png"):
                        cloud_file_path = os.path.join(
                            self.sat_dir, kaartblad_name, file
                        )
                        year, month, day = re.findall(
                            r"(\d{4})-(\d{1,2})-(\d{1,2})Z_cloud", file
                        )[0]
                        if year not in self.data_dict[kaartblad_name]["cloud_masks"]:
                            self.data_dict[kaartblad_name]["cloud_masks"][year] = {}

                        if (
                            month
                            not in self.data_dict[kaartblad_name]["cloud_masks"][year]
                        ):
                            self.data_dict[kaartblad_name]["cloud_masks"][year][
                                month
                            ] = {}
                        self.data_dict[kaartblad_name]["cloud_masks"][year][month][
                            day
                        ] = cloud_file_path

    def __len__(self):
        # Doesn't matter since the data is generated on the fly
        return 10000000

    def filter_by_year(self, years):
        for kaartblad in self.data_dict.keys():
            self.data_dict[kaartblad]["satellite_images"] = {
                year: value
                for year, value in self.data_dict[kaartblad]["satellite_images"].items()
                if year in years
            }
            self.data_dict[kaartblad]["cloud_masks"] = {
                year: value
                for year, value in self.data_dict[kaartblad]["cloud_masks"].items()
                if year in years
            }

    def filter_by_month(self, months):
        for kaartblad in self.data_dict.keys():
            for year in self.data_dict[kaartblad]["satellite_images"].keys():
                self.data_dict[kaartblad]["satellite_images"][year] = {
                    month: value
                    for month, value in self.data_dict[kaartblad]["satellite_images"][
                        year
                    ].items()
                    if month in months
                }
            for year in self.data_dict[kaartblad]["cloud_masks"].keys():
                self.data_dict[kaartblad]["cloud_masks"][year] = {
                    month: value
                    for month, value in self.data_dict[kaartblad]["cloud_masks"][
                        year
                    ].items()
                    if month in months
                }

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        while True:
            kaartbladen = list(self.data_dict.keys())
            kaartblad = random.choice(kaartbladen)
            years = list(self.data_dict[kaartblad]["satellite_images"].keys())
            year = random.choice(years)
            months = list(self.data_dict[kaartblad]["satellite_images"][year].keys())
            month = random.choice(months)
            days = list(
                self.data_dict[kaartblad]["satellite_images"][year][month].keys()
            )
            day = random.choice(days)

            gt_path = self.data_dict[kaartblad]["gt_path"]
            sat_path = self.data_dict[kaartblad]["satellite_images"][year][month][day]
            cloud_path = self.data_dict[kaartblad]["cloud_masks"][year][month][day]

            gt = load_tiff(gt_path)
            sat = load_tiff(sat_path)[:3]
            cloud_mask = np.array(Image.open(cloud_path))

            gtshp = gt.shape
            satshp = sat.shape
            cloudshp = cloud_mask.shape
            widths = [gtshp[1], satshp[1], cloudshp[0]]
            heights = [gtshp[2], satshp[2], cloudshp[1]]
            w_min = min(widths)
            h_min = min(heights)
            gt = gt[:, :w_min, :h_min]
            sat = sat[:, :w_min, :h_min]
            cloud_mask = cloud_mask[:w_min, :h_min]

            cloud_mask = np.expand_dims(cloud_mask, axis=0)
            cloud_mask = cloud_mask > 0
            nolabel_mask = gt == 0
            try:
                invalid_mask = np.logical_or(cloud_mask, nolabel_mask)
            except:
                invalid_mask = cloud_mask
                print(sat_path)

            # Normalize input data using linear normalization and cloud masking
            # Do this before any narrowing down so that we use the largest possible area to compute the histogram
            sat = satellite_normalization_with_cloud_masking(
                sat,
                cloud_mask,
                min_percentile=1,
                max_percentile=99,
                mask_value=1.0,
                norm_hi=self.norm_hi,
                norm_lo=self.norm_lo,
            )

            gt = torch.tensor(gt, dtype=torch.long)
            sat = torch.tensor(sat, dtype=torch.float32)
            valid_mask = torch.tensor(invalid_mask, dtype=torch.bool).logical_not()
            cloud_mask = torch.tensor(cloud_mask, dtype=torch.bool)
            label_mask = torch.tensor(nolabel_mask, dtype=torch.bool).logical_not()
            # Get a crop
            sat, gt, valid_mask, cloud_mask, label_mask = random_crop(
                sat,
                gt,
                valid_mask,
                cloud_mask,
                label_mask,
                self.patch_size,
                self.patch_size,
                padding_value_img=1,
                padding_value_gt=0,
                padding_value_valid_mask=0,
                padding_value_cloud_mask=1,
                padding_value_label_mask=0,
            )
            valid_ratio = (valid_mask.sum() / valid_mask.numel()).item()
            if valid_ratio > self.valid_threshold:
                break

        sample = {"gt": gt, "sat": sat, "valid_mask": valid_mask, "cloud_mask": cloud_mask, "label_mask": label_mask}
        return sample


class SatteliteValDataset(nn.Module):
    def __init__(
        self,
        root_dir,
        kaartbladen,
        years,
        months,
        patch_size=256,
        patch_offset=256,
        valid_threshold=0.5,
        norm_hi=None,
        norm_lo=None,
    ):
        """
        Arguments:
            root_dir (string): Directory with all the images.
                The structure of the root dir should be like:
                    root_dir/
                        data_gt\
                            gt_kaartblad_1.tiff
                            ...
                            gt_kaartblad_43.tiff
                        
                        data_sat\
                            kaartblad_1
                                kaartblad_1_202X-XX-XXZ.tif
                                ...
                                kaartblad_1_202X-XX-XXZ.tif
                            ...
                            kaartblad_43
                                kaartblad_43_202X-XX-XXZ.tif
                                ...
                                kaartblad_43_202X-XX-XXZ.tif
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.gt_dir = f"{root_dir}/val/data_gt"
        self.sat_dir = f"{root_dir}/val/data_sat"
        self.kaartbladen = kaartbladen
        self.kaartbladen_names = [f"kaartblad_{item}" for item in kaartbladen]
        self.years = years
        self.months = months

        self.patch_size = patch_size
        self.patch_offset = patch_offset
        self.norm_hi = norm_hi
        self.norm_lo = norm_lo
        self.valid_threshold = valid_threshold

        self.data_dict = {}
        self.build_data_dict()
        self.filter_by_year(years)
        self.filter_by_month(months)
        self.data_list = []
        self.build_data_list()

    def build_data_dict(self):
        print("Building the data dictionary...")
        for gt_file in os.listdir(self.gt_dir):
            gt_file_path = os.path.join(self.gt_dir, gt_file)
            kaartblad_name = re.findall(r"(kaartblad_\w+-\w).", gt_file)[0]
            if kaartblad_name in self.kaartbladen_names:
                self.data_dict[kaartblad_name] = {}
                self.data_dict[kaartblad_name]["gt_path"] = gt_file_path
                self.data_dict[kaartblad_name]["satellite_images"] = {}
                self.data_dict[kaartblad_name]["cloud_masks"] = {}
                for file in os.listdir(os.path.join(self.sat_dir, kaartblad_name)):
                    if file.endswith(".tif"):
                        sat_file_path = os.path.join(self.sat_dir, kaartblad_name, file)
                        year, month, day = re.findall(
                            r"(\d{4})-(\d{1,2})-(\d{1,2})Z", file
                        )[0]
                        if (
                            year
                            not in self.data_dict[kaartblad_name]["satellite_images"]
                        ):
                            self.data_dict[kaartblad_name]["satellite_images"][
                                year
                            ] = {}

                        if (
                            month
                            not in self.data_dict[kaartblad_name]["satellite_images"][
                                year
                            ]
                        ):
                            self.data_dict[kaartblad_name]["satellite_images"][year][
                                month
                            ] = {}

                        self.data_dict[kaartblad_name]["satellite_images"][year][month][
                            day
                        ] = sat_file_path
                    elif file.endswith(".png"):
                        cloud_file_path = os.path.join(
                            self.sat_dir, kaartblad_name, file
                        )
                        year, month, day = re.findall(
                            r"(\d{4})-(\d{1,2})-(\d{1,2})Z_cloud", file
                        )[0]
                        if year not in self.data_dict[kaartblad_name]["cloud_masks"]:
                            self.data_dict[kaartblad_name]["cloud_masks"][year] = {}

                        if (
                            month
                            not in self.data_dict[kaartblad_name]["cloud_masks"][year]
                        ):
                            self.data_dict[kaartblad_name]["cloud_masks"][year][
                                month
                            ] = {}
                        self.data_dict[kaartblad_name]["cloud_masks"][year][month][
                            day
                        ] = cloud_file_path

    def build_data_list(self):
        print("Building the data list...")
        for kaartblad in self.data_dict.keys():
            for year in self.data_dict[kaartblad]["satellite_images"].keys():
                for month in self.data_dict[kaartblad]["satellite_images"][year].keys():
                    for day in self.data_dict[kaartblad]["satellite_images"][year][
                        month
                    ].keys():
                        sample = {}
                        sample["gt_path"] = self.data_dict[kaartblad]["gt_path"]
                        sample["sat_path"] = self.data_dict[kaartblad][
                            "satellite_images"
                        ][year][month][day]
                        sample["cloud_path"] = self.data_dict[kaartblad]["cloud_masks"][
                            year
                        ][month][day]
                        self.data_list.append(sample)

    def __len__(self):
        return len(self.data_list)

    def filter_by_year(self, years):
        for kaartblad in self.data_dict.keys():
            self.data_dict[kaartblad]["satellite_images"] = {
                year: value
                for year, value in self.data_dict[kaartblad]["satellite_images"].items()
                if year in years
            }
            self.data_dict[kaartblad]["cloud_masks"] = {
                year: value
                for year, value in self.data_dict[kaartblad]["cloud_masks"].items()
                if year in years
            }

    def filter_by_month(self, months):
        for kaartblad in self.data_dict.keys():
            for year in self.data_dict[kaartblad]["satellite_images"].keys():
                self.data_dict[kaartblad]["satellite_images"][year] = {
                    month: value
                    for month, value in self.data_dict[kaartblad]["satellite_images"][
                        year
                    ].items()
                    if month in months
                }
            for year in self.data_dict[kaartblad]["cloud_masks"].keys():
                self.data_dict[kaartblad]["cloud_masks"][year] = {
                    month: value
                    for month, value in self.data_dict[kaartblad]["cloud_masks"][
                        year
                    ].items()
                    if month in months
                }

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data_list[idx]

        gt_path = sample["gt_path"]
        sat_path = sample["sat_path"]
        cloud_path = sample["cloud_path"]

        gt = load_tiff(gt_path)
        sat = load_tiff(sat_path)[:3]
        cloud_mask = np.array(Image.open(cloud_path))

        gtshp = gt.shape
        satshp = sat.shape
        cloudshp = cloud_mask.shape
        widths = [gtshp[1], satshp[1], cloudshp[0]]
        heights = [gtshp[2], satshp[2], cloudshp[1]]
        w_min = min(widths)
        h_min = min(heights)
        gt = gt[:, :w_min, :h_min]
        sat = sat[:, :w_min, :h_min]
        cloud_mask = cloud_mask[:w_min, :h_min]

        cloud_mask = np.expand_dims(cloud_mask, axis=0)
        cloud_mask = cloud_mask > 0
        nolabel_mask = gt == 0
        try:
            invalid_mask = np.logical_or(cloud_mask, nolabel_mask)
        except:
            print(gt_path, sat_path, cloud_path)

        # Normalize input data using linear normalization and cloud masking
        # Do this before any narrowing down so that we use the largest possible area to compute the histogram
        sat = satellite_normalization_with_cloud_masking(
            sat,
            cloud_mask,
            min_percentile=1,
            max_percentile=99,
            mask_value=1.0,
            norm_hi=self.norm_hi,
            norm_lo=self.norm_lo,
        )

        gt = torch.tensor(gt, dtype=torch.long)
        sat = torch.tensor(sat, dtype=torch.float32)
        valid_mask = torch.tensor(invalid_mask, dtype=torch.bool).logical_not()
        cloud_mask = torch.tensor(cloud_mask, dtype=torch.bool)
        label_mask = torch.tensor(nolabel_mask, dtype=torch.bool).logical_not()
        # Get a crop

        full_sat = copy.deepcopy(sat)
        full_gt = copy.deepcopy(gt)
        full_valid_mask = copy.deepcopy(valid_mask)
        full_cloud_mask = copy.deepcopy(cloud_mask)
        full_label_mask = copy.deepcopy(label_mask)

        sat = split_into_tiles(
            sat, tile_size=self.patch_size, offset=self.patch_offset, padding_value=1
        )
        gt = split_into_tiles(
            gt, tile_size=self.patch_size, offset=self.patch_offset, padding_value=0
        )
        valid_mask = split_into_tiles(
            valid_mask,
            tile_size=self.patch_size,
            offset=self.patch_offset,
            padding_value=0,
        )
        cloud_mask = split_into_tiles(
            cloud_mask,
            tile_size=self.patch_size,
            offset=self.patch_offset,
            padding_value=1,
        )
        label_mask = split_into_tiles(
            label_mask,
            tile_size=self.patch_size,
            offset=self.patch_offset,
            padding_value=0,
        )
        # Potentially only keep tiles with more than 50% valid mask ratio

        valid_indices = [
            (valid_mask_tile.sum() / valid_mask_tile.numel()).item()
            > self.valid_threshold
            for valid_mask_tile in valid_mask
        ]

        sat = [
            item if valid_indices[i] else torch.ones_like(item)
            for i, item in enumerate(sat)
        ]
        gt = [
            item if valid_indices[i] else torch.zeros_like(item)
            for i, item in enumerate(gt)
        ]
        valid_mask = [
            item if valid_indices[i] else torch.zeros_like(item)
            for i, item in enumerate(valid_mask)
        ]
        cloud_mask = [
            item if valid_indices[i] else torch.zeros_like(item)
            for i, item in enumerate(cloud_mask)
        ]
        label_mask = [
            item if valid_indices[i] else torch.zeros_like(item)
            for i, item in enumerate(label_mask)
        ]


        sat = torch.stack(sat)
        gt = torch.stack(gt)
        valid_mask = torch.stack(valid_mask)
        cloud_mask = torch.stack(cloud_mask)
        label_mask = torch.stack(label_mask)

        sample = {
            "gt": gt,
            "sat": sat,
            "valid_mask": valid_mask,
            "cloud_mask": cloud_mask,
            "label_mask": label_mask,
            "gt_full": full_gt,
            "sat_full": full_sat,
            "valid_mask_full": full_valid_mask,
            "valid_cloud_full": full_cloud_mask,
            "valid_label_full": full_label_mask,
        }
        # sample = {"gt": gt, "sat": sat, "valid_mask": valid_mask}

        return sample


class SatteliteTestDataset(nn.Module):
    def __init__(
            self,
            root_dir,
            kaartbladen,
            years,
            months,
            patch_size=256,
            patch_offset=256,
            valid_threshold=0.5,
            norm_hi=None,
            norm_lo=None,
    ):
        """
        Arguments:
            root_dir (string): Directory with all the images.
                The structure of the root dir should be like:
                    root_dir/
                        data_gt\
                            gt_kaartblad_1.tiff
                            ...
                            gt_kaartblad_43.tiff

                        data_sat\
                            kaartblad_1
                                kaartblad_1_202X-XX-XXZ.tif
                                ...
                                kaartblad_1_202X-XX-XXZ.tif
                            ...
                            kaartblad_43
                                kaartblad_43_202X-XX-XXZ.tif
                                ...
                                kaartblad_43_202X-XX-XXZ.tif
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        self.root_dir = root_dir
        self.gt_dir = f"{root_dir}/test/data_gt"
        self.sat_dir = f"{root_dir}/test/data_sat"
        self.kaartbladen = kaartbladen
        self.kaartbladen_names = [f"kaartblad_{item}" for item in kaartbladen]
        self.years = years
        self.months = months

        self.patch_size = patch_size
        self.patch_offset = patch_offset
        self.norm_hi = norm_hi
        self.norm_lo = norm_lo
        self.valid_threshold = valid_threshold

        self.data_dict = {}
        self.build_data_dict()
        self.filter_by_year(years)
        self.filter_by_month(months)
        self.data_list = []
        self.build_data_list()

    def build_data_dict(self):
        print("Building the data dictionary...")
        for gt_file in os.listdir(self.gt_dir):
            gt_file_path = os.path.join(self.gt_dir, gt_file)
            kaartblad_name = re.findall(r"(kaartblad_\w+-\w).", gt_file)[0]
            if kaartblad_name in self.kaartbladen_names:
                self.data_dict[kaartblad_name] = {}
                self.data_dict[kaartblad_name]["gt_path"] = gt_file_path
                self.data_dict[kaartblad_name]["satellite_images"] = {}
                self.data_dict[kaartblad_name]["cloud_masks"] = {}
                for file in os.listdir(os.path.join(self.sat_dir, kaartblad_name)):
                    if file.endswith(".tif"):
                        sat_file_path = os.path.join(self.sat_dir, kaartblad_name, file)
                        year, month, day = re.findall(
                            r"(\d{4})-(\d{1,2})-(\d{1,2})Z", file
                        )[0]
                        if (
                                year
                                not in self.data_dict[kaartblad_name]["satellite_images"]
                        ):
                            self.data_dict[kaartblad_name]["satellite_images"][
                                year
                            ] = {}

                        if (
                                month
                                not in self.data_dict[kaartblad_name]["satellite_images"][
                            year
                        ]
                        ):
                            self.data_dict[kaartblad_name]["satellite_images"][year][
                                month
                            ] = {}

                        self.data_dict[kaartblad_name]["satellite_images"][year][month][
                            day
                        ] = sat_file_path
                    elif file.endswith(".png"):
                        cloud_file_path = os.path.join(
                            self.sat_dir, kaartblad_name, file
                        )
                        year, month, day = re.findall(
                            r"(\d{4})-(\d{1,2})-(\d{1,2})Z_cloud", file
                        )[0]
                        if year not in self.data_dict[kaartblad_name]["cloud_masks"]:
                            self.data_dict[kaartblad_name]["cloud_masks"][year] = {}

                        if (
                                month
                                not in self.data_dict[kaartblad_name]["cloud_masks"][year]
                        ):
                            self.data_dict[kaartblad_name]["cloud_masks"][year][
                                month
                            ] = {}
                        self.data_dict[kaartblad_name]["cloud_masks"][year][month][
                            day
                        ] = cloud_file_path

    def build_data_list(self):
        print("Building the data list...")
        for kaartblad in self.data_dict.keys():
            for year in self.data_dict[kaartblad]["satellite_images"].keys():
                for month in self.data_dict[kaartblad]["satellite_images"][year].keys():
                    for day in self.data_dict[kaartblad]["satellite_images"][year][
                        month
                    ].keys():
                        sample = {}
                        sample["gt_path"] = self.data_dict[kaartblad]["gt_path"]
                        sample["sat_path"] = self.data_dict[kaartblad][
                            "satellite_images"
                        ][year][month][day]
                        sample["cloud_path"] = self.data_dict[kaartblad]["cloud_masks"][
                            year
                        ][month][day]
                        self.data_list.append(sample)

    def __len__(self):
        return len(self.data_list)

    def filter_by_year(self, years):
        for kaartblad in self.data_dict.keys():
            self.data_dict[kaartblad]["satellite_images"] = {
                year: value
                for year, value in self.data_dict[kaartblad]["satellite_images"].items()
                if year in years
            }
            self.data_dict[kaartblad]["cloud_masks"] = {
                year: value
                for year, value in self.data_dict[kaartblad]["cloud_masks"].items()
                if year in years
            }

    def filter_by_month(self, months):
        for kaartblad in self.data_dict.keys():
            for year in self.data_dict[kaartblad]["satellite_images"].keys():
                self.data_dict[kaartblad]["satellite_images"][year] = {
                    month: value
                    for month, value in self.data_dict[kaartblad]["satellite_images"][
                        year
                    ].items()
                    if month in months
                }
            for year in self.data_dict[kaartblad]["cloud_masks"].keys():
                self.data_dict[kaartblad]["cloud_masks"][year] = {
                    month: value
                    for month, value in self.data_dict[kaartblad]["cloud_masks"][
                        year
                    ].items()
                    if month in months
                }

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data_list[idx]

        gt_path = sample["gt_path"]
        sat_path = sample["sat_path"]
        cloud_path = sample["cloud_path"]

        gt = load_tiff(gt_path)
        sat = load_tiff(sat_path)[:3]
        cloud_mask = np.array(Image.open(cloud_path))

        gtshp = gt.shape
        satshp = sat.shape
        cloudshp = cloud_mask.shape
        widths = [gtshp[1], satshp[1], cloudshp[0]]
        heights = [gtshp[2], satshp[2], cloudshp[1]]
        w_min = min(widths)
        h_min = min(heights)
        gt = gt[:, :w_min, :h_min]
        sat = sat[:, :w_min, :h_min]
        cloud_mask = cloud_mask[:w_min, :h_min]

        cloud_mask = np.expand_dims(cloud_mask, axis=0)
        cloud_mask = cloud_mask > 0
        nolabel_mask = gt == 0
        try:
            invalid_mask = np.logical_or(cloud_mask, nolabel_mask)
        except:
            print(gt_path, sat_path, cloud_path)

        # Normalize input data using linear normalization and cloud masking
        # Do this before any narrowing down so that we use the largest possible area to compute the histogram
        sat = satellite_normalization_with_cloud_masking(
            sat,
            cloud_mask,
            min_percentile=1,
            max_percentile=99,
            mask_value=1.0,
            norm_hi=self.norm_hi,
            norm_lo=self.norm_lo,
        )

        gt = torch.tensor(gt, dtype=torch.long)
        sat = torch.tensor(sat, dtype=torch.float32)
        valid_mask = torch.tensor(invalid_mask, dtype=torch.bool).logical_not()
        cloud_mask = torch.tensor(cloud_mask, dtype=torch.bool)
        label_mask = torch.tensor(nolabel_mask, dtype=torch.bool).logical_not()
        # Get a crop

        full_sat = copy.deepcopy(sat)
        full_gt = copy.deepcopy(gt)
        full_valid_mask = copy.deepcopy(valid_mask)
        full_cloud_mask = copy.deepcopy(cloud_mask)
        full_label_mask = copy.deepcopy(label_mask)

        sat = split_into_tiles(
            sat, tile_size=self.patch_size, offset=self.patch_offset, padding_value=1
        )
        gt = split_into_tiles(
            gt, tile_size=self.patch_size, offset=self.patch_offset, padding_value=0
        )
        valid_mask = split_into_tiles(
            valid_mask,
            tile_size=self.patch_size,
            offset=self.patch_offset,
            padding_value=0,
        )
        cloud_mask = split_into_tiles(
            cloud_mask,
            tile_size=self.patch_size,
            offset=self.patch_offset,
            padding_value=1,
        )
        label_mask = split_into_tiles(
            label_mask,
            tile_size=self.patch_size,
            offset=self.patch_offset,
            padding_value=0,
        )
        # Potentially only keep tiles with more than 50% valid mask ratio

        valid_indices = [
            (valid_mask_tile.sum() / valid_mask_tile.numel()).item()
            > self.valid_threshold
            for valid_mask_tile in valid_mask
        ]

        sat = [
            item if valid_indices[i] else torch.ones_like(item)
            for i, item in enumerate(sat)
        ]
        gt = [
            item if valid_indices[i] else torch.zeros_like(item)
            for i, item in enumerate(gt)
        ]
        valid_mask = [
            item if valid_indices[i] else torch.zeros_like(item)
            for i, item in enumerate(valid_mask)
        ]
        cloud_mask = [
            item if valid_indices[i] else torch.zeros_like(item)
            for i, item in enumerate(cloud_mask)
        ]
        label_mask = [
            item if valid_indices[i] else torch.zeros_like(item)
            for i, item in enumerate(label_mask)
        ]


        sat = torch.stack(sat)
        gt = torch.stack(gt)
        valid_mask = torch.stack(valid_mask)
        cloud_mask = torch.stack(cloud_mask)
        label_mask = torch.stack(label_mask)

        sample = {
            "gt": gt,
            "sat": sat,
            "valid_mask": valid_mask,
            "cloud_mask": cloud_mask,
            "label_mask": label_mask,
            "gt_full": full_gt,
            "sat_full": full_sat,
            "valid_mask_full": full_valid_mask,
            "valid_cloud_full": full_cloud_mask,
            "valid_label_full": full_label_mask,
        }
        # sample = {"gt": gt, "sat": sat, "valid_mask": valid_mask}

        return sample