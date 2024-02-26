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


def ImageResampler_Liver(filepath, name, gt):
    img = sitk.ReadImage(filepath)
    target_sapce = [1.3333, 1.3333, 1.3333]
    new_size = [96, 96, 96]

    if gt:
        interpolater = sitk.sitkNearestNeighbor
    else:
        interpolater = sitk.sitkBSpline
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)
    resampler.SetOutputSpacing(target_sapce)
    resampler.SetInterpolator(interpolater)

    resampler.SetSize(new_size)
    resampled_img = resampler.Execute(img)
    sitk.WriteImage(resampled_img, name)


def ImageResampler_BraTS(filepath, name, gt):
    # 240, 240, 155
    image = sitk.ReadImage(filepath)

    # 图像填充到(160, 240, 240)
    # 计算在每个方向上需要添加的层数
    pad_before = [0, 0, 3]  # 在图像前面填充的层数
    pad_after = [0, 0, 2]  # 在图像后面填充的层数，这里添加5层到第一维

    # 使用Pad函数进行填充
    padded_image = sitk.ConstantPad(image, pad_before, pad_after, constant=0)  # 假定用0值填充

    # 计算新的spacing
    new_size = [96, 96, 64]
    original_size = padded_image.GetSize()
    original_spacing = padded_image.GetSpacing()

    new_spacing = [
        (original_size[0] * original_spacing[0]) / new_size[0],
        (original_size[1] * original_spacing[1]) / new_size[1],
        (original_size[2] * original_spacing[2]) / new_size[2],
    ]

    # 创建重新采样器
    if gt:
        interpolater = sitk.sitkNearestNeighbor
    else:
        interpolater = sitk.sitkLinear
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)

    resampler.SetInterpolator(interpolater)
    resampler.SetDefaultPixelValue(0)  # 背景像素设为0

    # 应用重新采样
    resampled_image = resampler.Execute(padded_image)

    sitk.WriteImage(resampled_image, name)


def Deform_BraTS(filepath):
    """
    gaussian = GaussianFilter(3, self.sigma, 3.0).to(device=_device)
    offset = torch.as_tensor(self.rand_offset, device=_device).unsqueeze(0)
    grid[:3] += gaussian(offset)[0] * self.magnitude
    """
    img_list = []
    for file in os.listdir(filepath):
        if not file.endswith("elastic.nii.gz"):
            if file.endswith("seg.nii.gz"):
                lab = sitk.ReadImage(os.path.join(filepath, file))
                # lab = sitk.GetArrayFromImage(lab).astype(np.int8)[None, ...]
            else:
                img_list.append(sitk.ReadImage(os.path.join(filepath, file)))
                # img_list.append(sitk.GetArrayFromImage(tmp).astype(np.float32)[None, ...])
    imgA, imgB, imgC, imgD = img_list

    rand_elastic_transform = Rand3DElasticd(
        keys=['imageA', 'imageB', 'imageC', 'imageD', 'label'],  # 字典中图像和标签的键名
        mode=('bilinear', 'bilinear', 'bilinear', 'bilinear', 'nearest'),  # 分别为图像和mask设置插值方式
        prob=1.0,  # 变换应用的概率
        sigma_range=(8, 15),  # 高斯核的标准差范围
        magnitude_range=(100, 300),  # 弹性变形的幅度范围
        spatial_size=(155, 240, 240),  # 图像的空间大小
    )
    # rand_elastic_transform.set_random_state(seed=0)
    transformed = rand_elastic_transform({
        'imageA': sitk.GetArrayFromImage(imgA).astype(np.int16)[None, ...],
        'imageB': sitk.GetArrayFromImage(imgB).astype(np.int16)[None, ...],
        'imageC': sitk.GetArrayFromImage(imgC).astype(np.int16)[None, ...],
        'imageD': sitk.GetArrayFromImage(imgD).astype(np.int16)[None, ...],
        'label': sitk.GetArrayFromImage(lab).astype(np.float32)[None, ...]
    })
    transformed_imageA = sitk.GetImageFromArray(transformed['imageA'].squeeze())
    transformed_imageA.CopyInformation(imgA)
    transformed_imageB = sitk.GetImageFromArray(transformed['imageB'].squeeze())
    transformed_imageB.CopyInformation(imgB)
    transformed_imageC = sitk.GetImageFromArray(transformed['imageC'].squeeze())
    transformed_imageC.CopyInformation(imgC)
    transformed_imageD = sitk.GetImageFromArray(transformed['imageD'].squeeze())
    transformed_imageD.CopyInformation(imgD)

    transformed_mask = sitk.GetImageFromArray(transformed['label'].squeeze())
    transformed_mask.CopyInformation(lab)
    sitk.WriteImage(transformed_imageA, filepath + "img-t1c_elastic.nii.gz")
    sitk.WriteImage(transformed_imageB, filepath + "img-t1n_elastic.nii.gz")
    sitk.WriteImage(transformed_imageC, filepath + "img-t2f_elastic.nii.gz")
    sitk.WriteImage(transformed_imageD, filepath + "img-t2w_elastic.nii.gz")
    sitk.WriteImage(transformed_mask, filepath + "seg_elastic.nii.gz")

    deform_dice = np.mean(Metrics.dice_BraTS(transformed['label'].squeeze(),
                                             sitk.GetArrayFromImage(lab).astype(np.float32))[::3])
    print(deform_dice)
    return deform_dice
    # ssim = Loss.SSIM3D(kernel_size=9)


def get_BB(mainpath):
    bool_img = np.ones((155, 240, 240), dtype=bool)
    for case in os.listdir(mainpath):
        print(case)
        for img in os.listdir(os.path.join(mainpath, case)):
            img = sitk.ReadImage(os.path.join(mainpath, case, img))
            img = sitk.GetArrayFromImage(img)
            # 将灰度值大于0的部分设置为False，小于等于0的部分设置为True
            tmp = img <= 0
            # 初始化结果张量为第一个图像的布尔张量
            bool_img = np.logical_and(bool_img, tmp)

    # 找到所有True值的坐标
    true_positions = np.argwhere(bool_img == False)

    # 计算沿每个维度的最小和最大索引
    min_coords = true_positions.min(axis=0)
    max_coords = true_positions.max(axis=0)

    # 创建包围盒坐标
    # 由于索引是从0开始的，我们需要在最大坐标上加1来确保包含所有的True值
    bounding_box = np.array([min_coords, max_coords + 1])

    print(f"包围盒坐标: {bounding_box}")


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

    valid = ["patient101", "patient102", "patient103", "patient104", "patient105",
             "patient106", "patient107", "patient108", "patient109", "patient110"]

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


def preprocess_Liver():
    """
    {
    "image_ED": "./toy_sample/Patient2/fixed.nii.gz",
    "image_ES": "./toy_sample/Patient2/moving.nii.gz",
    "label_ED": "./toy_sample/Patient2/fixed_gt.nii.gz",
    "label_ES": "./toy_sample/Patient2/moving_gt.nii.gz"
    }
    """
    # path = "../../datasets/combine"
    # path = "../../datasets/lspig"
    # path = "../../datasets/lits"
    path = "../../datasets/sliver/testing"
    file_list = []
    file_json = []
    # for file in os.listdir(path + "/img"):
    #     if file.endswith(".nii.gz"):
    #         ImageResampler_Liver(os.path.join(path + "/img", file),
    #                              os.path.join(path + "/img", file.split(".nii.gz")[0] + "_size96.nii.gz"),
    #                              False)
    # for file in os.listdir(path + "/mask"):
    #     if file.endswith(".nii.gz"):
    #         ImageResampler_Liver(os.path.join(path + "/mask", file),
    #                              os.path.join(path + "/mask", file.split(".nii.gz")[0] + "_size96.nii.gz"),
    #                              True)

    for f in os.listdir(path + "/img"):
        for m in os.listdir(path + "/img"):
            cond = f.endswith("_size96.nii.gz") and m.endswith("_size96.nii.gz")
            if cond and not f == m:
                cfg_dict = {}
                cfg_dict["image_fixed"] = os.path.join("img", f)
                cfg_dict["image_moving"] = os.path.join("img", m)
                label_path = os.path.join("mask", f.split("_size96.nii.gz")[0] + "_mask_size96.nii.gz")
                assert os.path.exists(os.path.join(path, label_path))
                cfg_dict["label_fixed"] = label_path
                label_path = os.path.join("mask", m.split("_size96.nii.gz")[0] + "_mask_size96.nii.gz")
                assert os.path.exists(os.path.join(path, label_path))
                cfg_dict["label_moving"] = label_path

                assert len(cfg_dict) == 4
                if np.random.rand() < 0.1:
                    file_json.append(cfg_dict)
    print(len(file_json))
    with open(os.path.join(path, "val.json"), "w", encoding="utf-8") as f:
        json.dump(file_json, f, indent=4, separators=(",", ":"), ensure_ascii=False)
    # with open(os.path.join(path, "train.json"), "w", encoding="utf-8") as f:
    #     json.dump(file_json, f, indent=4, separators=(",", ":"), ensure_ascii=False)


def preprocess_BraTS_local():
    path = "../BraTS_test/"
    target = "../BraTS_t2wM-t1nF/testing"

    # valid = ["patient101", "patient102", "patient103", "patient104", "patient105",
    #          "patient106", "patient107", "patient108", "patient109", "patient110"]

    file_json = []
    for folder in os.listdir(path):
        if not folder.endswith(".json"):
            #  and folder in valid
            print(folder)
            casepath = os.path.join(path, folder)
            targetpath = os.path.join(target, folder)
            if not os.path.exists(targetpath):
                os.mkdir(targetpath)

            cfg_dict = {}

            for file in os.listdir(casepath):
                if "fixed" not in file and "moving" not in file:
                    if file.endswith("t1n.nii.gz"):
                        ImageResampler_BraTS(os.path.join(casepath, file), os.path.join(targetpath, "fixed.nii.gz"),
                                             False)
                        cfg_dict["image_fixed"] = os.path.join(folder, "fixed.nii.gz")
                    if file.endswith("seg.nii.gz"):
                        ImageResampler_BraTS(os.path.join(casepath, file), os.path.join(targetpath, "fixed_gt.nii.gz"),
                                             True)
                        cfg_dict["label_fixed"] = os.path.join(folder, "fixed_gt.nii.gz")
                    if file.endswith("t2w.nii.gz"):
                        ImageResampler_BraTS(os.path.join(casepath, file), os.path.join(targetpath, "moving.nii.gz"),
                                             False)
                        cfg_dict["image_moving"] = os.path.join(folder, "moving.nii.gz")
                    if file.endswith("seg.nii.gz"):
                        ImageResampler_BraTS(os.path.join(casepath, file), os.path.join(targetpath, "moving_gt.nii.gz"),
                                             True)
                        cfg_dict["label_moving"] = os.path.join(folder, "moving_gt.nii.gz")

            assert len(cfg_dict) == 4
            file_json.append(cfg_dict)
    # with open(os.path.join(path, "test.json"), "w", encoding="utf-8") as f:
    #     json.dump(file_json, f, indent=4, separators=(",", ":"), ensure_ascii=False)
    # with open(os.path.join(path, "train.json"), "w", encoding="utf-8") as f:
    #     json.dump(file_json, f, indent=4, separators=(",", ":"), ensure_ascii=False)


def preprocess_BraTS_server():
    path = "../../datasets/BraTS_t2wM-t1nF/testing"
    # path = "../../datasets/BraTS_t2wM-t1nF/testing"

    # valid = ["patient101", "patient102", "patient103", "patient104", "patient105",
    #          "patient106", "patient107", "patient108", "patient109", "patient110"]

    file_json = []
    for folder in os.listdir(path):
        if not folder.endswith(".json"):
            #  and folder in valid
            print(folder)
            casepath = os.path.join(path, folder)
            # targetpath = os.path.join(target, folder)
            # if not os.path.exists(targetpath):
            #     os.mkdir(targetpath)

            cfg_dict = {}

            for file in os.listdir(casepath):
                if file.endswith("fixed.nii.gz"):
                    # ImageResampler_BraTS(os.path.join(casepath, file), os.path.join(targetpath, "fixed.nii.gz"),
                    #                      False)
                    cfg_dict["image_fixed"] = os.path.join(folder, "fixed.nii.gz")
                if file.endswith("fixed_gt.nii.gz"):
                    # ImageResampler_BraTS(os.path.join(casepath, file), os.path.join(targetpath, "fixed_gt.nii.gz"),
                    #                      True)
                    cfg_dict["label_fixed"] = os.path.join(folder, "fixed_gt.nii.gz")
                if file.endswith("moving.nii.gz"):
                    # ImageResampler_BraTS(os.path.join(casepath, file), os.path.join(targetpath, "moving.nii.gz"),
                    #                      False)
                    cfg_dict["image_moving"] = os.path.join(folder, "moving.nii.gz")
                if file.endswith("moving_gt.nii.gz"):
                    # ImageResampler_BraTS(os.path.join(casepath, file), os.path.join(targetpath, "moving_gt.nii.gz"),
                    #                      True)
                    cfg_dict["label_moving"] = os.path.join(folder, "moving_gt.nii.gz")

            assert len(cfg_dict) == 4
            file_json.append(cfg_dict)
    with open(os.path.join(path, "test.json"), "w", encoding="utf-8") as f:
        json.dump(file_json, f, indent=4, separators=(",", ":"), ensure_ascii=False)
    # with open(os.path.join(path, "train.json"), "w", encoding="utf-8") as f:
    #     json.dump(file_json, f, indent=4, separators=(",", ":"), ensure_ascii=False)


if __name__ == "__main__":
    # get_BB("../BraTS_train/")

    preprocess_BraTS_local()

    # for case in os.listdir("../BraTS_test/"):
    #     print(case)
    #     deform_dice = 0.0
    #     cnt = 0
    #     while (deform_dice <= 0.7 or deform_dice >= 0.99) and cnt <= 2:
    #         deform_dice = Deform_BraTS("../BraTS_test/" + case + "/")
    #         cnt += 1
