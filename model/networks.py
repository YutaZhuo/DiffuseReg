import functools
import logging
import torch
import torch.nn as nn
from torch.nn import init

logger = logging.getLogger('base')


####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    logger.info('Initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(
            weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt):
    self_opt = opt['self']
    model_opt = opt['model']
    from .diffusion_3D import CFG_diffusion, unet, uvit, transmorph, swin_unet, \
        swin_unetR, swin_unetR_uvit, swin_unetR_cat

    # if 'dual' in self_opt or 'vmdiff' in self_opt:
    if model_opt["type"] == "unet":
        print("with unet")
        model_score = unet.myUNet(
            in_channel=model_opt['unet']['in_channel'],
            out_channel=model_opt['unet']['out_channel'],
            inner_channel=model_opt['unet']['inner_channel'],
            channel_mults=model_opt['unet']['channel_multiplier'],
            attn_res=model_opt['unet']['attn_res'],
            res_blocks=model_opt['unet']['res_blocks'],
            dropout=model_opt['unet']['dropout'],
            image_size=model_opt['diffusion']['image_size'],
            opt=self_opt
        )
    elif model_opt["type"] == "uvit":
        print("with uvit")
        model_score = uvit.UViT(
            img_size=model_opt['uvit']['img_size'],
            in_chans=model_opt['uvit']['in_chans'],
            out_chans=model_opt['uvit']['out_chans'],
            patch_size=model_opt['uvit']['patch_size'],
            embed_dim=model_opt['uvit']['embed_dim'],
            depth=model_opt['uvit']['depth'],
            num_heads=model_opt['uvit']['num_heads'],
            mlp_ratio=model_opt['uvit']['mlp_ratio'],
            qkv_bias=model_opt['uvit']['qkv_bias'],
            mlp_time_embed=model_opt['uvit']['mlp_time_embed'],
            cond=model_opt['uvit']['cond']
        )
    elif model_opt["type"] == "transmorph":
        print("with transmorph")
        model_score = transmorph.TransMorph(
            **model_opt['transmorph'],
        )
    elif model_opt["type"] == "swin_unet":
        print("with swin unet")
        model_score = swin_unet.SwinUnet(
            **model_opt['swin_unet'],
        )
    elif model_opt["type"] == "swin_unetR":
        print("with swin unetR")
        model_score = swin_unetR.SwinUNETR(
            **model_opt['swin_unetR'],
        )
    elif model_opt["type"] == "swin_unetR_uvit":
        print("with swin unetR uvit")
        model_score = swin_unetR_uvit.SwinUNETR(
            **model_opt['swin_unetR_uvit'],
        )
    elif model_opt["type"] == "swin_unetR_cat":
        print("with swin unetR cat")
        model_score = swin_unetR_cat.SwinUNETR(
            **model_opt['swin_unetR_cat'],
        )

    else:
        raise NotImplementedError(model_opt["type"])

    stn = unet.SpatialTransform(model_opt['diffusion']['image_size'])

    bootstrap = unet.RecursiveCascadeNetwork(
        n_cascades=model_opt['bootstrap']['n_cas'],
        im_size=model_opt['diffusion']['image_size'],
        network=model_opt['bootstrap']['module'],
        stn=stn)
    print("bootstrap loading checkpoing:", model_opt['bootstrap']['checkpoint'])
    params_dict = torch.load(model_opt['bootstrap']['checkpoint'])
    for i, submodel in enumerate(bootstrap.stems):
        submodel.load_state_dict(params_dict["cascade {}".format(i)])
    bootstrap.eval()

    netG = CFG_diffusion.GaussianDiffusion(
        model_score, stn, bootstrap,
        channels=model_opt['diffusion']['channels'],
        loss_type='l2',  # L1 or L2
        conditional=model_opt['diffusion']['conditional'],
        schedule_opt=model_opt['beta_schedule']['train'],
        loss_lambda=model_opt['loss_lambda'],
        gamma=model_opt['gamma'],
        scaler=model_opt['diffusion']['flow_scaler'],
        mean=model_opt['diffusion']['flow_mean'],
        std=model_opt['diffusion']['flow_std']
    )

    if opt['phase'] == 'train':
        load_path = opt['path']['resume_state']
        if load_path is None:
            if model_opt["type"] == "unet":
                init_weights(netG.denoise_fn, init_type='orthogonal')
            elif model_opt["type"] == "uvit":
                pass
            elif model_opt["type"] == "transmorph":
                pass
            elif model_opt["type"] == "swin_unet":
                pass
            elif model_opt["type"] in ["swin_unetR", "swin_unetR_uvit", "swin_unetR_cat"]:
                pass
            else:
                raise NotImplementedError(model_opt["type"])
            init_weights(netG.stn, init_type='normal')
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG
