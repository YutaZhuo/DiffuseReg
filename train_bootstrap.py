import argparse
import os
import torch
import time
from model.diffusion_3D.unet import RecursiveCascadeNetwork, SpatialTransform
from model.diffusion_3D.loss import loss_RCN

import core.logger as Logger
import data as Data
from math import *

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

import numpy as np


def main(args):
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    writer = SummaryWriter(opt['path']["tb_logger"])

    # dataset
    phase = 'train'
    finesize = opt['model']['diffusion']['image_size']
    dataset_opt = opt['datasets']['train']
    batchSize = opt['datasets']['train']['batch_size']
    train_set = Data.create_dataset_Brats(dataset_opt, finesize, phase)
    train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
    training_iters = int(ceil(train_set.data_len / float(batchSize)))
    print('Dataset Initialized')

    reconstruction = SpatialTransform(finesize).cuda()
    model = RecursiveCascadeNetwork(n_cascades=opt['model']['bootstrap']['n_cas'],
                                    im_size=finesize,
                                    network=opt['model']['bootstrap']['module'],
                                    stn=reconstruction).cuda()
    n_epoch = opt['train']['n_epoch']
    print("{} cascades VTN".format(opt['model']['bootstrap']['n_cas']))
    if args.finetune:
        print("load checkpoint")
        params_dict = torch.load(opt['model']['bootstrap']['checkpoint'])
        for i, submodel in enumerate(model.stems):
            submodel.load_state_dict(params_dict["cascade {}".format(i)])

    if args.strategy == "plain":
        trainable_params = []
        for submodel in model.stems:
            trainable_params += list(submodel.parameters())

        optim = Adam(trainable_params, lr=1e-4)
    else:
        raise NotImplementedError

    cnter = 1
    for epoch in range(1, n_epoch + 1):
        print(f"-----Epoch {epoch} / {n_epoch}-----")
        print(f">>>>> Train:")

        if args.strategy == "plain":
            model.train()
            # print(len(train_loader))
            t0 = time.perf_counter()
            rec_lossL = []
            sim_lossL = []
            for istep, train_data in enumerate(train_loader):

                fixed, moving = train_data["F"].cuda(), train_data["M"].cuda()
                flows, warps, results = model(fixed, moving)
                rec_loss, sim_loss = loss_RCN(results, None, fixed)

                optim.zero_grad()
                rec_loss.backward()
                optim.step()

                writer.add_scalar(tag="Loss/reconstruction",
                                  scalar_value=rec_loss.item(),
                                  global_step=cnter)
                writer.add_scalar(tag="Loss/SSIM",
                                  scalar_value=sim_loss.item(),
                                  global_step=cnter)

                cnter += 1
                rec_lossL.append(rec_loss.item())
                sim_lossL.append(sim_loss.item())
            print("Rec Loss: {}".format(round(np.array(rec_lossL).mean(), 4)))
            print("Sim Loss: {}".format(round(np.array(sim_lossL).mean(), 4)))
            t1 = time.perf_counter()
            print("train time: {}".format(round(t1 - t0), 2))
        else:
            raise NotImplementedError
    # scheduler.step()

    ckp = {}
    for i, submodel in enumerate(model.stems):
        ckp[f"cascade {i}"] = submodel.state_dict()
    ckp['epoch'] = n_epoch
    torch.save(ckp, "{}/E{}.pth".format(opt['path']['checkpoint'], n_epoch))


if __name__ == '__main__':
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    parser = argparse.ArgumentParser()

    parser.add_argument('-g', "--gpu_ids", type=str,
                        default="0")
    parser.add_argument("--strategy", type=str,
                        default="plain")
    parser.add_argument('-c', '--config', type=str,
                        default='config/train_VTN.json')
    parser.add_argument('--finetune', action="store_true")

    args = parser.parse_args()

    main(args)
