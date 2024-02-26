import SimpleITK as sitk
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import re
from monai.transforms import (Compose, Rand3DElasticd, RandRotate90d, RandAxisFlipd)
import core.metrics as Metrics
import model.diffusion_3D.loss as Loss


def ImageResampler_ACDC(filepath, name, gt):
    img = sitk.ReadImage(filepath)
    space = img.GetSpacing()
    dim = img.GetSize()

    target_sapce = (1.5, 1.5, 3.15)
    if gt:
        interpolater = sitk.sitkNearestNeighbor
    else:
        interpolater = sitk.sitkBSpline
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing(target_sapce)
    resampler.SetInterpolator(interpolater)
    new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc in
                zip(dim, space, target_sapce)]
    resampler.SetSize(new_size)
    resampled_img = resampler.Execute(img)
    sitk.WriteImage(resampled_img, name)


def preprocess_ACDC():
    """
    {
    "image_ED": "./toy_sample/Patient2/fixed.nii.gz",
    "image_ES": "./toy_sample/Patient2/moving.nii.gz",
    "label_ED": "./toy_sample/Patient2/fixed_gt.nii.gz",
    "label_ES": "./toy_sample/Patient2/moving_gt.nii.gz"
    }
    """
    path = "../../datasets/ACDC/database/testing"
    # path = "../../datasets/ACDC/database/training"

    valid = []

    file_json = []
    for folder in os.listdir(path):
        if not folder.endswith(".json"):
            #  and folder in valid
            print(folder)
            casepath = os.path.join(path, folder)
            ED = None
            ES = None
            cfg_dict = {}
            with open(os.path.join(casepath, "Info.cfg")) as cfg:
                # ED-fixed, ES-moving
                for line in cfg.readlines():
                    if "ED" in line:
                        ED = int(re.findall(r'\d+', line)[0])
                    elif "ES" in line:
                        ES = int(re.findall(r'\d+', line)[0])
            for file in os.listdir(casepath):
                if not file.endswith(".cfg") and ("fixed" not in file and "moving" not in file):
                    if file.endswith("frame{:02}.nii.gz".format(ED)):
                        # ImageResampler(os.path.join(casepath, file), os.path.join(casepath, "fixed.nii.gz"), False)
                        cfg_dict["image_ED"] = os.path.join(folder, "fixed.nii.gz")
                    if file.endswith("frame{:02}_gt.nii.gz".format(ED)):
                        # ImageResampler(os.path.join(casepath, file), os.path.join(casepath, "fixed_gt.nii.gz"), True)
                        cfg_dict["label_ED"] = os.path.join(folder, "fixed_gt.nii.gz")
                    if file.endswith("frame{:02}.nii.gz".format(ES)):
                        # ImageResampler(os.path.join(casepath, file), os.path.join(casepath, "moving.nii.gz"), False)
                        cfg_dict["image_ES"] = os.path.join(folder, "moving.nii.gz")
                    if file.endswith("frame{:02}_gt.nii.gz".format(ES)):
                        # ImageResampler(os.path.join(casepath, file), os.path.join(casepath, "moving_gt.nii.gz"), True)
                        cfg_dict["label_ES"] = os.path.join(folder, "moving_gt.nii.gz")

            assert len(cfg_dict) == 4
            file_json.append(cfg_dict)
    with open(os.path.join(path, "test50.json"), "w", encoding="utf-8") as f:
        json.dump(file_json, f, indent=4, separators=(",", ":"), ensure_ascii=False)
    # with open(os.path.join(path, "train.json"), "w", encoding="utf-8") as f:
    #     json.dump(file_json, f, indent=4, separators=(",", ":"), ensure_ascii=False)




if __name__ == "__main__":
    preprocess_ACDC()

