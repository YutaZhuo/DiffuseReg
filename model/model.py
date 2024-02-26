from collections import OrderedDict
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from . import metrics as Metrics


class DDPM(BaseModel):
    def __init__(self, opt):
        super(DDPM, self).__init__(opt)
        # define network and load pretrained models
        self.netG = self.set_device(networks.define_G(opt))

        self.schedule_phase = None
        self.centered = opt['datasets']['centered']

        ######################## check num of params here with DEBUG    ########################
        """
        from thop import profile, clever_format
        batch_time = torch.full((1,), 0.001).cuda()
        flops, prarms = profile(self.netG.denoise_fn, inputs=(torch.rand((1, 5, 32, 128, 128)).cuda(), batch_time))
        flops, params = clever_format([flops, prarms], "%.3f")
        print(flops, params)
        """
        ########################                                        ########################

        # set loss and load resume state
        self.set_loss()
        self.set_new_noise_schedule(opt['model']['beta_schedule']['train'], schedule_phase='train')

        if self.opt['phase'] == 'train':
            self.netG.train()
            # find the parameters to optimize
            if opt['model']['finetune_norm']:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find('transformer') >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)

            else:
                optim_params = list(self.netG.parameters())

            if opt['train']["optimizer"]["type"] == "adam":
                self.optG = torch.optim.Adam(optim_params, lr=opt['train']["optimizer"]["lr"])
            elif opt['train']["optimizer"]["type"] == "adamw":
                self.optG = torch.optim.AdamW(
                    optim_params,
                    lr=opt['train']["optimizer"]["lr"],
                    weight_decay=opt['train']["optimizer"]["weight_decay"])
            else:
                raise NotImplementedError(opt['train']["optimizer"]["type"])

            if opt['train']["optimizer"]["schedule"] == "cosine":
                self.scheduler = torch.optim.lr_scheduler \
                    .CosineAnnealingLR(self.optG,
                                       eta_min=opt['train']["optimizer"]["eta_min"],
                                       T_max=opt['train']["n_epoch"])
            elif opt['train']["optimizer"]["schedule"] == "cosinewr":
                self.scheduler = torch.optim.lr_scheduler \
                    .CosineAnnealingWarmRestarts(self.optG,
                                                 eta_min=opt['train']["optimizer"]["eta_min"],
                                                 T_0=opt['train']["n_epoch"] / 4,
                                                 T_mult=3)
            elif opt['train']["optimizer"]["schedule"] == "plain":
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optG, lambda e: 1)
            else:
                raise NotImplementedError(opt['train']["optimizer"]["schedule"])

            self.log_dict = OrderedDict()
        self.load_network()
        # self.scaler = torch.cuda.amp.GradScaler()
        if self.opt[self.opt['phase']]["amp"]:
            print("AMP ON")
        else:
            print("AMP OFF")

        # self.print_network(self.netG)

    def feed_data(self, data):
        self.data = self.set_device(data)

    def optimize_parameters(self):
        if self.opt[self.opt['phase']]["amp"]:
            # self.optG.zero_grad()
            # with torch.cuda.amp.autocast():
            #     loss = self.netG(self.data)
            # # need to average in multi-gpu
            #
            # # l_tot = loss
            # diff_loss, sim_loss, reg_loss, total_loss = loss
            # self.scaler.scale(total_loss).backward()
            # # total_loss.backward()
            # self.scaler.step(self.optG)
            # # self.optG.step()
            # self.scaler.update()
            raise NotImplementedError(self.opt["train"]["amp"])
        else:
            self.optG.zero_grad()
            loss = self.netG(self.data)

            # l_tot = loss
            diff_loss, sim_loss, reg_loss, total_loss = loss
            total_loss.backward()
            self.optG.step()
        # set log
        self.log_dict['l_dif'] = diff_loss.item()
        self.log_dict['l_sim'] = sim_loss.item()
        self.log_dict['l_reg'] = reg_loss.item()
        self.log_dict['l_tot'] = total_loss.item()

    def test_registration(self):
        self.netG.eval()
        with torch.no_grad():
            if self.opt[self.opt['phase']]["amp"]:
                # with torch.cuda.amp.autocast():
                #     input = torch.cat([self.data['M'], self.data['F']], dim=1)
                #     if isinstance(self.netG, nn.DataParallel):
                #         self.out_M, self.flow, self_contD, self.contF = self.netG.module.registration(input)
                #     else:
                #         self.out_M, self.flow, self.contD, self.contF = self.netG.registration(input)
                raise NotImplementedError(self.opt[self.opt['phase']]["amp"])
            else:
                input = torch.cat([self.data['M'], self.data['F']], dim=1)
                if isinstance(self.netG, nn.DataParallel):
                    self.out_M, self.flow, self_contD, self.contF = self.netG.module.registration(input)
                else:
                    self.out_M, self.flow, self.contD, self.contF = self.netG.registration(input)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, nn.DataParallel):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_new_noise_schedule(
                    schedule_opt, self.device)
            else:
                self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_registration(self):
        out_dict = OrderedDict()

        out_dict['warp'] = self.out_M.detach()
        out_dict['flow'] = self.flow.detach()
        out_dict['contD'] = self.contD
        out_dict['contF'] = self.contF
        out_dict['flow_visual'] = Metrics.tensor2im(self.flow.detach().float().cpu(), min_max=(-1, 1))
        return out_dict

    def save_network(self, epoch, iter_step):
        genG_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen_G.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, genG_path)

        # opt
        opt_state = {'epoch': epoch, 'iter': iter_step,
                     'scheduler': None, 'optimizer': None}
        opt_state['optimizer'] = self.optG.state_dict()
        torch.save(opt_state, opt_path)

    def load_network(self):
        load_path = self.opt['path']['resume_state']

        if load_path is not None:
            print(load_path)
            genG_path = '{}_gen_G.pth'.format(load_path)

            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                genG_path), strict=(not self.opt['model']['finetune_norm']))

            if self.opt['phase'] == 'train':
                # optimizer
                opt = torch.load(opt_path)
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']
