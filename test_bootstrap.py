import argparse
import os
import torch
from timeit import default_timer
from model.diffusion_3D.unet import RecursiveCascadeNetwork, SpatialTransform
from model.diffusion_3D.loss import loss_RCN

import core.logger as Logger
import core.metrics as Metrics
import data as Data
from math import *

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
import time
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


def main(args):
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    writer = SummaryWriter(opt['path']["tb_logger"])

    # dataset
    phase = 'test'
    finesize = opt['model']['diffusion']['image_size']
    dataset_opt = opt['datasets']['test']
    test_set = Data.create_dataset_ACDC(dataset_opt, finesize, "test50")
    test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
    print('Dataset Initialized')

    reconstruction = SpatialTransform(finesize).cuda()
    model = RecursiveCascadeNetwork(n_cascades=opt['model']['bootstrap']['n_cas'],
                                    im_size=finesize,
                                    network=opt['model']['bootstrap']['module'],
                                    stn=reconstruction).cuda()
    params_dict = torch.load(args.weights)
    # model.stems = [torch.nn.DataParallel(submodel) for submodel in model.stems]
    for i, submodel in enumerate(model.stems):
        submodel.load_state_dict(params_dict["cascade {}".format(i)])

    registDice = np.zeros((len(test_set), 3))
    originDice = np.zeros((len(test_set), 3))
    registTime = []
    print('Begin Model Evaluation.')
    model.eval()
    print(len(test_loader))
    flow_xL = []
    flow_yL = []
    flow_zL = []

    for istep, test_data in enumerate(test_loader):
        t0 = default_timer()
        with torch.no_grad():
            fixed, moving = test_data["F"].cuda(), test_data["M"].cuda()
            fixed = fixed.squeeze().unsqueeze(0).unsqueeze(0)
            moving = moving.squeeze().unsqueeze(0).unsqueeze(0)

            flows, warps, results = model(fixed, moving)
            flow = flows[-1]

        t1 = default_timer()
        origin_seg = test_data['MS'].squeeze().unsqueeze(0).unsqueeze(0).cuda()
        regist_seg = reconstruction(origin_seg.type(torch.float32), flow, mode="nearest")
        label_seg = test_data['FS'].squeeze().unsqueeze(0).unsqueeze(0).cuda()
        regist_img = reconstruction(moving.type(torch.float32), flow, mode="bilinear")

        tmp_MS = sitk.GetImageFromArray(origin_seg.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_MS, "./toy_sample/VTN_origin_seg.nii.gz")
        tmp_M = sitk.GetImageFromArray(moving.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_M, "./toy_sample/VTN_origin.nii.gz")
        tmp_WS = sitk.GetImageFromArray(regist_seg.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_WS, "./toy_sample/VTN_regist_seg.nii.gz")
        tmp_W = sitk.GetImageFromArray(regist_img.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_W, "./toy_sample/VTN_regist.nii.gz")
        tmp_FS = sitk.GetImageFromArray(label_seg.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_FS, "./toy_sample/VTN_label_seg.nii.gz")
        tmp_F = sitk.GetImageFromArray(fixed.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_F, "./toy_sample/VTN_label.nii.gz")
        flow_vis = sitk.GetImageFromArray(flow.detach().squeeze().permute(1, 2, 3, 0).cpu().numpy())
        sitk.WriteImage(flow_vis, f"./toy_sample/VTN_flow.nii.gz")


        vals_regist = Metrics.dice_BraTS(regist_seg.cpu().numpy(), label_seg.cpu().numpy())[::3]
        vals_origin = Metrics.dice_BraTS(origin_seg.cpu().numpy(), label_seg.cpu().numpy())[::3]
        registDice[istep] = vals_regist
        originDice[istep] = vals_origin
        print('---- Original Dice: %03f | Deformed Dice: %03f' % (np.mean(vals_origin), np.mean(vals_regist)))

        # vals_regist = Metrics.mask_metrics(regist_seg, label_seg)
        # vals_origin = Metrics.mask_metrics(origin_seg, label_seg)
        # registDice[istep] = vals_regist.item()
        # originDice[istep] = vals_origin.item()
        # print('---- Original Dice: %03f | Deformed Dice: %03f' % (vals_origin, vals_regist))

        # vals_regist = Metrics.dice_ACDC(regist_seg.cpu().numpy(), label_seg.cpu().numpy())[::3]
        # vals_origin = Metrics.dice_ACDC(origin_seg.cpu().numpy(), label_seg.cpu().numpy())[::3]
        # registDice[istep] = vals_regist
        # originDice[istep] = vals_origin
        # print('---- Original Dice: %03f | Deformed Dice: %03f' % (np.mean(vals_origin), np.mean(vals_regist)))

        flow_np = flow.squeeze().cpu().numpy()
        flow_x = flow_np[0]
        flow_y = flow_np[1]
        flow_z = flow_np[2]
        # print(flow_x.mean(), flow_x.std(), flow_x.min(), flow_x.max())
        # print(flow_y.mean(), flow_y.std(), flow_y.min(), flow_y.max())
        # print(flow_z.mean(), flow_z.std(), flow_z.min(), flow_z.max())

        flow_xL.append(flow_x.flatten())
        flow_yL.append(flow_y.flatten())
        flow_zL.append(flow_z.flatten())

        time.sleep(1)
    omdice, osdice = np.mean(originDice), np.std(originDice)
    mdice, sdice = np.mean(registDice), np.std(registDice)
    flow_xL = np.concatenate(flow_xL, axis=None)
    flow_yL = np.concatenate(flow_yL, axis=None)
    flow_zL = np.concatenate(flow_zL, axis=None)

    print()
    print(flow_xL.mean(), flow_xL.std(), flow_xL.min(), flow_xL.max())  # 0.0014  0.4636 -14.0123 10.3754
    print(flow_yL.mean(), flow_yL.std(), flow_yL.min(), flow_yL.max())  # -0.0758 1.1375 -15.7695 16.4528
    print(flow_zL.mean(), flow_zL.std(), flow_zL.min(), flow_zL.max())  # -0.1493 1.2221 -18.4860 15.3057
    # plt.hist(flow_xL, bins=50, label="x")
    # plt.legend()
    # plt.show()
    # plt.hist(flow_yL, bins=50, label="y")
    # plt.legend()
    # plt.show()
    # plt.hist(flow_zL, bins=50, label="z")
    # plt.legend()
    # plt.show()

    print('---------------------------------------------')
    print('Total Dice and Time Metrics------------------')
    print('---------------------------------------------')
    print('origin Dice | mean = %.3f, std= %.3f' % (omdice, osdice))
    print(f'origin detailed Dice | mean = {np.mean(originDice, axis=0)}({np.std(originDice, axis=0)})')
    print('Deform Dice | mean = %.3f, std= %.3f' % (mdice, sdice))
    print(f'Deform detailed Dice | mean = {np.mean(registDice, axis=0)}({np.std(registDice, axis=0)})')


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    parser = argparse.ArgumentParser()

    parser.add_argument('-g', "--gpu_ids", type=str,
                        default="1")
    parser.add_argument("--strategy", type=str,
                        default="plain")
    parser.add_argument('-c', '--config', type=str,
                        default='config/test_VTN.json')
    parser.add_argument('-w', '--weights', type=str,
                        default="./experiments/.../checkpoint/E2000.pth")
    args = parser.parse_args()

    main(args)
