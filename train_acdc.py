import os
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
import os
from math import *
import time
from torch.utils import tensorboard
import numpy as np
import GPUwaiter
from model.diffusion_3D.unet import SpatialTransform
import SimpleITK as sitk

if __name__ == "__main__":
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/train_3D.json',
                        help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default="1")
    # parse configs
    args = parser.parse_args()
    if args.gpu_ids == "wait":
        args.gpu_ids = GPUwaiter.waiter()

    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    writer = tensorboard.SummaryWriter(opt['path']["tb_logger"])
    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # dataset
    phase = 'train'
    finesize = opt['model']['diffusion']['image_size']
    dataset_opt = opt['datasets']['train']
    batchSize = opt['datasets']['train']['batch_size']
    train_set = Data.create_dataset_ACDC(dataset_opt, finesize, phase)
    train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
    training_iters = int(ceil(train_set.data_len / float(batchSize)))
    print('Dataset Initialized')

    # model
    diffusion = Model.create_model(opt)
    print("Model Initialized")

    # Train

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']
    if opt['path']['resume_state']:
        print('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

    cnter = 0
    while current_epoch < n_epoch:
        current_epoch += 1
        for istep, train_data in enumerate(train_loader):
            iter_start_time = time.time()
            current_step += 1
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            t = (time.time() - iter_start_time) / batchSize
            # log
            message = '(epoch: %d | iters: %d/%d | time: %.3f) ' % (current_epoch, (istep + 1), training_iters, t)
            errors = diffusion.get_current_log()
            for k, v in errors.items():
                message += '%s: %.6f ' % (k, v)
            print(message)
            if (istep + 1) % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                t = (time.time() - iter_start_time) / batchSize
                writer.add_scalar("train/l_dif", logs['l_dif'], cnter)
                writer.add_scalar("train/l_sim", logs['l_sim'], cnter)
                writer.add_scalar("train/l_reg", logs['l_reg'], cnter)
                writer.add_scalar("train/l_tot", logs['l_tot'], cnter)
                cnter += 1
        diffusion.scheduler.step()
        writer.add_scalar("lr", diffusion.scheduler.get_last_lr()[0], current_epoch)

        if current_epoch in opt['train']['val_freq']:
            testdataset_opt = {
                "name": "3D",
                "dataroot": "../datasets/ACDC/database/testing",
                "data_len": 3
            }
            test_set = Data.create_dataset_ACDC(testdataset_opt, finesize, "test")
            test_loader = Data.create_dataloader(test_set, testdataset_opt, "test")
            stn = SpatialTransform(finesize).cuda()
            registDice = np.zeros((len(test_set), 5))
            originDice = np.zeros((len(test_set), 5))
            registSSIM = np.zeros(len(test_set))
            originSSIM = np.zeros(len(test_set))
            for istep, test_data in enumerate(test_loader):
                diffusion.feed_data(test_data)
                diffusion.test_registration()
                visuals = diffusion.get_current_registration()
                # print(visuals['contF'].shape)
                flow = visuals["flow"]
                warp = visuals["warp"]
                moving_seg = test_data['MS'].squeeze().unsqueeze(0).unsqueeze(0).cuda()
                regist_seg = stn(moving_seg.type(torch.float32), flow, mode="nearest")
                fixed_seg = test_data['FS'].squeeze().unsqueeze(0).unsqueeze(0).cuda()

                moving = test_data['M'].squeeze().unsqueeze(0).unsqueeze(0).cuda()
                fixed = test_data['F'].squeeze().unsqueeze(0).unsqueeze(0).cuda()
                regist = stn(moving.type(torch.float32), flow)

                # tmp_WS = sitk.GetImageFromArray(regist_seg.squeeze().cpu().numpy())
                # sitk.WriteImage(tmp_WS, f"./toy_sample/regist_seg_{current_epoch}_{istep}.nii.gz")
                # tmp_W = sitk.GetImageFromArray(regist.squeeze().cpu().numpy())
                # sitk.WriteImage(tmp_W, f"./toy_sample/regist_{current_epoch}_{istep}.nii.gz")
                # flow_vis = sitk.GetImageFromArray(flow.detach().squeeze().permute(1, 2, 3, 0).cpu().numpy())
                # sitk.WriteImage(flow_vis, f"./toy_sample/flow_{current_epoch}_{istep}.nii.gz")

                vals_regist = Metrics.dice_ACDC(regist_seg.cpu().numpy(), fixed_seg.cpu().numpy())[::3]
                vals_origin = Metrics.dice_ACDC(moving_seg.cpu().numpy(), fixed_seg.cpu().numpy())[::3]

                ssim_regist = round(diffusion.netG.loss_ssim(regist, fixed).item(), 4)
                ssim_origin = round(diffusion.netG.loss_ssim(moving, fixed).item(), 4)

                registDice[istep] = vals_regist
                originDice[istep] = vals_origin
                registSSIM[istep] = ssim_regist
                originSSIM[istep] = ssim_origin


                time.sleep(1)

            writer.add_scalar("eval/dice", np.mean(registDice), current_epoch)
            writer.add_scalar("eval/ssim", np.mean(registSSIM), current_epoch)

        if current_epoch in opt['train']['save_checkpoint_epoch'] or current_epoch == n_epoch:
            diffusion.save_network(current_epoch, current_step)
