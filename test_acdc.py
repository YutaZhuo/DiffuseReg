import os
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from math import *
import time
import numpy as np
import torch.nn.functional as F
from model.diffusion_3D.unet import SpatialTransform
import SimpleITK as sitk

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/test_3D.json',
                        help='JSON file for configuration')
    # swin_unetR_aug+1.0sim@adamw[1e-4,1e-4]cosine[1e-6]_train_240128_135907
    parser.add_argument('-w', '--weights', type=str,
                        default='./experiments/swin_unetR_aug+1.0sim+0.1reg@adamw[1e-4,1e-4]cosine[1e-6]_train_240221_154334/checkpoint/I120000_E1200',
                        help='weights file for validation')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default='0')

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    phase = 'test'
    finesize = opt['model']['diffusion']['image_size']
    dataset_opt = opt['datasets']['test']
    test_set = Data.create_dataset_ACDC(dataset_opt, finesize, "vis")
    test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
    print('Dataset Initialized')

    opt['path']['resume_state'] = args.weights
    # model
    diffusion = Model.create_model(opt)
    stn = SpatialTransform(finesize).cuda()
    print("Model Initialized")
    # Train

    registDice = np.zeros((len(test_set), 5))
    originDice = np.zeros((len(test_set), 5))
    registJacc = np.zeros((len(test_set), 5))
    originJacc = np.zeros((len(test_set), 5))
    registSSIM = np.zeros(len(test_set))
    originSSIM = np.zeros(len(test_set))
    NJD = np.zeros(len(test_set))
    JSD = np.zeros(len(test_set))

    registTime = []
    print('Begin Model Evaluation.')
    idx_ = 0
    result_path = '{}'.format(opt['path']['results'])

    os.makedirs(result_path, exist_ok=True)
    print(len(test_loader))

    for istep, test_data in enumerate(test_loader):
        idx_ += 1
        dataName = istep
        time1 = time.time()
        diffusion.feed_data(test_data)
        diffusion.test_registration()
        time2 = time.time()
        visuals = diffusion.get_current_registration()
        flow = visuals["flow"]
        # warp = visuals["warp"]
        moving_seg = test_data['MS'].squeeze().unsqueeze(0).unsqueeze(0).cuda()
        regist_seg = stn(moving_seg.type(torch.float32), flow, mode="nearest")
        fixed_seg = test_data['FS'].squeeze().unsqueeze(0).unsqueeze(0).cuda()

        moving = test_data['M'].squeeze().unsqueeze(0).unsqueeze(0).cuda()
        fixed = test_data['F'].squeeze().unsqueeze(0).unsqueeze(0).cuda()
        regist = stn(moving.type(torch.float32), flow)

        # save Intermediate results of the sampling process
        for idx, item in enumerate(visuals['contF']):
            flow_vis = sitk.GetImageFromArray(item.detach().squeeze().permute(1, 2, 3, 0).cpu().numpy())
            sitk.WriteImage(flow_vis, f"./toy_sample/x_start{istep}_{idx}.nii.gz")
            regist_tmp = stn(moving.type(torch.float32), item)
            regist_tmp_vis = sitk.GetImageFromArray(regist_tmp.squeeze().cpu().numpy())
            sitk.WriteImage(regist_tmp_vis, f"./toy_sample/regist{istep}_{idx}.nii.gz")

        tmp_MS = sitk.GetImageFromArray(moving_seg.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_MS, f"./toy_sample/moving_seg{istep}.nii.gz")
        tmp_M = sitk.GetImageFromArray(moving.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_M, f"./toy_sample/moving{istep}.nii.gz")
        tmp_WS = sitk.GetImageFromArray(regist_seg.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_WS, f"./toy_sample/regist_seg{istep}.nii.gz")
        tmp_W = sitk.GetImageFromArray(regist.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_W, f"./toy_sample/regist{istep}.nii.gz")
        tmp_FS = sitk.GetImageFromArray(fixed_seg.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_FS, f"./toy_sample/fixed_seg{istep}.nii.gz")
        tmp_F = sitk.GetImageFromArray(fixed.squeeze().cpu().numpy())
        sitk.WriteImage(tmp_F, f"./toy_sample/fixed{istep}.nii.gz")
        flow_vis = sitk.GetImageFromArray(flow.detach().squeeze().permute(1, 2, 3, 0).cpu().numpy())
        sitk.WriteImage(flow_vis, f"./toy_sample/flow{istep}.nii.gz")

        vals_regist = Metrics.dice_ACDC(regist_seg.cpu().numpy(), fixed_seg.cpu().numpy())[::3]
        vals_origin = Metrics.dice_ACDC(moving_seg.cpu().numpy(), fixed_seg.cpu().numpy())[::3]
        jacc_regist = Metrics.jacc_ACDC(regist_seg.cpu().numpy(), fixed_seg.cpu().numpy())[::3]
        jacc_origin = Metrics.jacc_ACDC(moving_seg.cpu().numpy(), fixed_seg.cpu().numpy())[::3]
        # vals_regist = Metrics.mask_metrics(regist_seg, fixed_seg).item()
        # vals_origin = Metrics.mask_metrics(moving_seg, fixed_seg).item()
        # jacc_regist = -1
        # jacc_origin = -1

        ssim_regist = round(diffusion.netG.loss_ssim(regist, fixed).item(), 4)
        ssim_origin = round(diffusion.netG.loss_ssim(moving, fixed).item(), 4)
        njd_flow = Metrics.neg_jacobian_det(flow).item()
        jsd_flow = Metrics.jacobian_det_var(flow).item()

        registDice[istep] = vals_regist
        originDice[istep] = vals_origin
        registJacc[istep] = jacc_regist
        originJacc[istep] = jacc_origin
        registSSIM[istep] = ssim_regist
        originSSIM[istep] = ssim_origin
        NJD[istep] = njd_flow
        JSD[istep] = jsd_flow
        print('---- Original Dice: %03f | Deformed Dice: %03f' % (np.mean(vals_origin), np.mean(vals_regist)))
        print('---- Original Jacc: %03f | Deformed Jacc: %03f' % (np.mean(jacc_origin), np.mean(jacc_regist)))
        print('---- Original SSIM: %03f | Deformed SSIM: %03f' % (ssim_origin, ssim_regist))
        print('---- NJD: %03f | JSD: %03f' % (njd_flow, jsd_flow))

        registTime.append(time2 - time1)
        time.sleep(1)
    omdice, osdice = np.mean(originDice), np.std(originDice)
    mdice, sdice = np.mean(registDice), np.std(registDice)
    omjacc, osjacc = np.mean(originJacc), np.std(originJacc)
    mjacc, sjacc = np.mean(registJacc), np.std(registJacc)

    mtime, stime = np.mean(registTime), np.std(registTime)
    omssim, osssim = np.mean(originSSIM), np.std(originSSIM)
    mssim, sssim = np.mean(registSSIM), np.std(registSSIM)
    mnjd, snjd = np.mean(NJD), np.std(NJD)
    mjsd, sjsd = np.mean(JSD), np.std(JSD)

    print()
    print('---------------------------------------------')
    print('Total Dice and Time Metrics------------------')
    print('---------------------------------------------')
    print('origin Dice | mean = %.3f, std= %.3f' % (omdice, osdice))
    print(f'origin detailed Dice | mean = {np.mean(originDice, axis=0)}({np.std(originDice, axis=0)})')
    print('Deform Dice | mean = %.3f, std= %.3f' % (mdice, sdice))
    print(f'Deform detailed Dice | mean = {np.mean(registDice, axis=0)}({np.std(registDice, axis=0)})')

    print('origin Jacc | mean = %.3f, std= %.3f' % (omjacc, osjacc))
    print(f'origin detailed Jacc | mean = {np.mean(originJacc, axis=0)}({np.std(originJacc, axis=0)})')
    print('Deform Jacc | mean = %.3f, std= %.3f' % (mjacc, sjacc))
    print(f'Deform detailed Jacc | mean = {np.mean(registJacc, axis=0)}({np.std(registJacc, axis=0)})')

    print('Deform Time | mean = %.3f, std= %.3f' % (mtime, stime))
    print('origin SSIM | mean = %.3f, std= %.3f' % (omssim, osssim))
    print('regist SSIM | mean = %.3f, std= %.3f' % (mssim, sssim))
    print('NJD | mean = %.3f, std= %.3f' % (mnjd, snjd))
    print('JSD | mean = %.3f, std= %.3f' % (mjsd, sjsd))

    # registDice = registDice.mean(axis=1)
    # originDice = originDice.mean(axis=1)
    # registJacc = registJacc.mean(axis=1)
    # originJacc = originJacc.mean(axis=1)
    # stack = np.column_stack([originDice, originJacc, originSSIM,
    #                          registDice, registJacc, registSSIM,
    #                          NJD, JSD])
    # np.savetxt(f"./toy_sample/1.0sim+0.1reg_ACDC_results.csv", stack, delimiter=',', fmt='%f')