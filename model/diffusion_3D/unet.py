import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from einops import rearrange


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def convolve(in_channels, out_channels, kernel_size, stride, dim=3):
    # through verification, the padding surely is 1, as input_size is even, kernel=3, stride=1 or 2
    return nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=1)


def convolveLeakyReLU(in_channels, out_channels, kernel_size, stride, dim=3):
    # the seq of conv and activation is reverse to origin paper
    return nn.Sequential(nn.LeakyReLU(0.1), convolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def upconvolve(in_channels, out_channels, kernel_size, stride, dim=3):
    # through verification, the padding surely is 1
    return nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride, padding=1)


def upconvolveLeakyReLU(in_channels, out_channels, kernel_size, stride, dim=3):
    # the seq of conv and activation is reverse to origin paper
    return nn.Sequential(nn.LeakyReLU(0.1), upconvolve(in_channels, out_channels, kernel_size, stride, dim=dim))


def affine_flow(W, b, sd, sh, sw):
    # W: (B, 3, 3), b: (B, 3), len = 128
    device = W.device
    b = b.view([-1, 3, 1, 1, 1])

    xr = torch.arange(-(sw - 1) / 2.0, sw / 2.0, 1.0, dtype=torch.float32)
    xr = xr.view([1, 1, 1, 1, -1]).to(device)
    yr = torch.arange(-(sh - 1) / 2.0, sh / 2.0, 1.0, dtype=torch.float32)
    yr = yr.view([1, 1, 1, -1, 1]).to(device)
    zr = torch.arange(-(sd - 1) / 2.0, sd / 2.0, 1.0, dtype=torch.float32)
    zr = zr.view([1, 1, -1, 1, 1]).to(device)

    wx = W[:, :, 0]
    wx = wx.view([-1, 3, 1, 1, 1])
    wy = W[:, :, 1]
    wy = wy.view([-1, 3, 1, 1, 1])
    wz = W[:, :, 2]
    wz = wz.view([-1, 3, 1, 1, 1])
    return xr * wx + yr * wy + zr * wz + b


# model
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000) / dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.inv_freq)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv3d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=4, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv3d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, dropout=0):
        super().__init__()
        self.mlp = nn.Sequential(
            Swish(),
            nn.Linear(time_emb_dim, dim_out)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out, dropout=dropout)
        self.res_conv = nn.Conv3d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        h = self.block1(x)
        if exists(self.mlp):
            h += self.mlp(time_emb)[:, :, None, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=4):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(4, in_channel)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, depth, height, width = input.shape
        n_head = self.n_head

        norm = self.norm(input)
        # print(norm.shape)
        qkv = self.qkv(norm)
        # print(qkv.shape)
        q, k, v = rearrange(qkv, 'b (qkv heads c) d h w -> qkv b heads c (d h w)', heads=n_head, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (d h w) -> b (heads c) d h w', heads=n_head, h=height, w=width)
        out = self.out(out)
        return out + input


class EfficientAttention(nn.Module):
    def __init__(self, in_channel, n_head=4):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(4, in_channel)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, depth, height, width = input.shape
        n_head = self.n_head

        norm = self.norm(input)
        # print(norm.shape)
        qkv = self.qkv(norm).chunk(3, dim=1)
        # print(qkv.shape)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h=n_head), qkv)
        k = k.softmax(dim=-1)
        q = q.softmax(dim=-2)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b h c (x y z) -> b (h c) x y z', h=n_head, x=depth, y=height, z=width)
        out = self.out(out)
        return out + input


class Attention(nn.Module):
    def __init__(self, in_channel, n_head=4):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(4, in_channel)
        self.qkv = nn.Conv3d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv3d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, depth, height, width = input.shape
        n_head = self.n_head

        norm = self.norm(input)
        # print(norm.shape)
        qkv = self.qkv(norm).chunk(3, dim=1)
        # print(qkv.shape)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y z -> b h c (x y z)', h=n_head), qkv)
        sim = torch.einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = torch.einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y z) d -> b (h d) x y z', h=n_head, x=depth, y=height, z=width)

        out = self.out(out)
        return out + input


class SelfAttention_fuse(nn.Module):
    def __init__(self, in_channel, n_head=4):
        super().__init__()

        self.n_head = n_head
        self.norm = nn.GroupNorm(4, in_channel)
        self.out = nn.Conv3d(in_channel, in_channel, 1)
        self.defmgen = nn.Conv3d(in_channel, 3, 3, padding=1)
        self.nonlinear = nn.Conv3d(3, 3, 3, padding=1)

    def forward(self, q, k, v, size):
        batch, channel, depth, height, width = q.shape

        n_head = self.n_head
        residual = q
        norm_q = self.norm(q)
        norm_k = self.norm(k)
        norm_v = self.norm(v)

        qkv = torch.cat([norm_q, norm_k, norm_v], dim=1)
        q, k, v = rearrange(qkv, 'b (qkv heads c) d h w -> qkv b heads c (d h w)', heads=n_head, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (d h w) -> b (heads c) d h w', heads=n_head, h=height, w=width)
        out = self.out(out)
        out = self.defmgen(out + residual)
        out = F.upsample_nearest(out, size)
        out = self.nonlinear(out)
        return out


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0, with_attn=None):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, time_emb_dim, dropout=dropout)
        if with_attn == "Attention":
            self.attn = Attention(dim_out)
        elif with_attn == "EfficientAttention":
            self.attn = EfficientAttention(dim_out)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if (self.with_attn) != "None":
            x = self.attn(x)
        return x


class UNet(nn.Module):
    def __init__(
            self,
            in_channel=6,
            out_channel=3,
            inner_channel=32,
            channel_mults=(1, 2, 4, 8, 8),
            attn_res=(8),
            res_blocks=3,
            dropout=0,
            with_time_emb=True,
            image_size=128,
            opt=None
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None
        self.opt = opt
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size[1]
        downs = [nn.Conv3d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel,
                               time_emb_dim=time_dim, dropout=dropout, with_attn=False)
        ])

        ups_diff = []
        ups_regis = []
        ups_adapt = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                feat_channel = feat_channels.pop()
                ups_diff.append(ResnetBlocWithAttn(
                    pre_channel + feat_channel, channel_mult, time_emb_dim=time_dim, dropout=dropout,
                    with_attn=use_attn))
                regischannel = pre_channel + feat_channel + channel_mult
                ups_regis.append(ResnetBlocWithAttn(
                    regischannel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))
                ups_adapt.append(
                    SelfAttention_fuse(channel_mult)
                )
                pre_channel = channel_mult
            if not is_last:
                ups_adapt.append(nn.Identity())
                ups_diff.append(Upsample(pre_channel))
                ups_regis.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups_diff = nn.ModuleList(ups_diff)
        self.ups_regis = nn.ModuleList(ups_regis)
        self.ups_adapt = nn.ModuleList(ups_adapt)
        self.final_conv = Block(pre_channel, default(out_channel, in_channel))
        # self.final_attn=SelfAttention_fuse(1)
        self.final_conv_defm = Block(pre_channel + 1, 3, groups=3)

    def forward(self, x, x_m, time):
        input_size = (x.size(2), x.size(3), x.size(4))
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
        x_1 = x
        x_2 = x
        defm = []
        # x_1vis=[]
        for layerd, layerr, layera in zip(self.ups_diff, self.ups_regis, self.ups_adapt):
            if isinstance(layerd, ResnetBlocWithAttn):
                feat = feats.pop()
                x_1 = layerd(torch.cat((x_1, feat), dim=1), t)
                x_2 = layerr(torch.cat((x_2, feat, x_1), dim=1), t)
                defm_ = layera(x_2, x_1, x_1, input_size)
                defm.append(defm_)
            else:
                x_1 = layerd(x_1)
                x_2 = layerr(x_2)
        recon = self.final_conv(x_1)
        defm = torch.stack(defm, dim=1)
        defm = torch.cat([defm, self.final_conv_defm(torch.cat((x_2, recon), dim=1)).unsqueeze_(1)], dim=1)
        defm = torch.mean(defm, dim=1)
        return recon, defm


class myUNet(nn.Module):
    def __init__(
            self,
            in_channel=5,
            out_channel=3,
            inner_channel=32,
            channel_mults=(1, 2, 4, 8, 8),
            attn_res=None,
            res_blocks=3,
            dropout=0,
            with_time_emb=True,
            image_size=None,
            opt=None
    ):
        super().__init__()

        if with_time_emb:
            time_dim = inner_channel
            self.time_mlp = nn.Sequential(
                TimeEmbedding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            time_dim = None
            self.time_mlp = None
        self.opt = opt
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size[0]
        downs = [nn.Conv3d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = attn_res[ind]
            if use_attn != "None":
                print("Downsample using atten res:", now_res, attn_res[ind])
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, time_emb_dim=time_dim, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, time_emb_dim=time_dim,
                               dropout=dropout, with_attn="Attention"),
            ResnetBlocWithAttn(pre_channel, pre_channel,
                               time_emb_dim=time_dim, dropout=dropout, with_attn="None")
        ])

        ups_diff = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = attn_res[ind]
            if use_attn != "None":
                print("Upsample using atten res:", now_res, attn_res[ind])
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                feat_channel = feat_channels.pop()
                ups_diff.append(ResnetBlocWithAttn(
                    pre_channel + feat_channel, channel_mult, time_emb_dim=time_dim, dropout=dropout,
                    with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups_diff.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups_diff = nn.ModuleList(ups_diff)
        self.final_conv = Block(pre_channel, out_channel)

    def forward_with_cond_scale(self,
                                x, time,
                                cond_scale=1.,
                                rescaled_phi=0.,
                                **kwargs):
        """
        x ch: [M, F, (flow noise)]
        """
        guided_flow = self.forward(x, time)
        if cond_scale == 1:
            return guided_flow

        b, _, h, w, d = x.shape
        random_guide = torch.randn(size=(1, 2, h, w, d)).repeat(b, 1, 1, 1, 1).cuda()
        x[:, [0, 1]] = random_guide
        null_flow = self.forward(x, time)
        scaled_flow = null_flow + (guided_flow - null_flow) * cond_scale
        if rescaled_phi == 0.:
            return scaled_flow

    def forward(self, x, time):
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
        x_1 = x
        for layerd in self.ups_diff:
            if isinstance(layerd, ResnetBlocWithAttn):
                feat = feats.pop()
                x_1 = layerd(torch.cat((x_1, feat), dim=1), t)
            else:
                x_1 = layerd(x_1)
        recon = self.final_conv(x_1)
        return recon


class VTNAffineStemlowRes(nn.Module):

    def __init__(self, dim=3, channels=16, flow_multiplier=1., im_size=None):
        super(VTNAffineStemlowRes, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.dim = dim
        self.im_size = im_size

        # Network architecture
        # The first convolution's input is the concatenated image
        self.conv1 = convolveLeakyReLU(2, channels, 3, 2)
        self.conv2 = convolveLeakyReLU(channels, 2 * channels, 3, 2)
        self.conv3 = convolveLeakyReLU(2 * channels, 4 * channels, 3, 2)
        self.conv3_1 = convolveLeakyReLU(4 * channels, 4 * channels, 3, 1)
        self.conv4 = convolveLeakyReLU(4 * channels, 8 * channels, 3, 2)
        self.conv4_1 = convolveLeakyReLU(8 * channels, 8 * channels, 3, 1)

        # ks = (2, 8, 8)  # ks is set depending on the output of conv6_1
        ks = (4, 6, 6)
        self.conv7_W = nn.Conv3d(8 * channels, 9, ks, 1, bias=False)
        self.conv7_b = nn.Conv3d(8 * channels, 3, ks, 1, bias=False)

    def forward(self, fixed, moving):
        concat_image = torch.cat((fixed, moving), dim=1)
        x1 = self.conv1(concat_image)  # C16, 64
        x2 = self.conv2(x1)  # C32, 32
        x3 = self.conv3(x2)  # C64, 16
        x3_1 = self.conv3_1(x3)  # C64, 16
        x4 = self.conv4(x3_1)  # C128, 8
        x4_1 = self.conv4_1(x4)  # C128, 8

        x7_W = self.conv7_W(x4_1)  # (B, 9, 1, 1, 1)
        x7_b = self.conv7_b(x4_1)  # (B, 3, 1, 1, 1)

        W = torch.reshape(x7_W, [-1, 3, 3]) * self.flow_multiplier
        b = torch.reshape(x7_b, [-1, 3]) * self.flow_multiplier
        sx, sy, sz = self.im_size
        flow = affine_flow(W, b, sx, sy, sz)  # flow: (B, 3, 128, 128, 128)

        return {
            'flow': flow,
            'W': W,
            'b': b,
        }


class VTNlowRes(nn.Module):
    def __init__(self, dim=3, flow_multiplier=1., channels=16, im_size=None):
        super(VTNlowRes, self).__init__()
        self.flow_multiplier = flow_multiplier
        self.channels = channels
        self.dim = dim

        # Network architecture
        # The first convolution's input is the concatenated image
        self.conv1 = convolveLeakyReLU(2, channels, 3, 2, dim=dim)
        self.conv2 = convolveLeakyReLU(channels, 2 * channels, 3, 2, dim=dim)
        self.conv3 = convolveLeakyReLU(2 * channels, 4 * channels, 3, 2, dim=dim)
        self.conv3_1 = convolveLeakyReLU(4 * channels, 4 * channels, 3, 1, dim=dim)
        self.conv4 = convolveLeakyReLU(4 * channels, 8 * channels, 3, 2, dim=dim)
        self.conv4_1 = convolveLeakyReLU(8 * channels, 8 * channels, 3, 1, dim=dim)

        self.pred4 = convolve(8 * channels, dim, 3, 1, dim=dim)  # 258 = 64 * channels + 1 + 1
        self.upsamp4to3 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv3 = upconvolveLeakyReLU(8 * channels, 4 * channels, 4, 2, dim=dim)

        self.pred3 = convolve(8 * channels + dim, dim, 3, 1, dim=dim)
        self.upsamp3to2 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv2 = upconvolveLeakyReLU(8 * channels + dim, 2 * channels, 4, 2, dim=dim)

        self.pred2 = convolve(4 * channels + dim, dim, 3, 1, dim=dim)
        self.upsamp2to1 = upconvolve(dim, dim, 4, 2, dim=dim)
        self.deconv1 = upconvolveLeakyReLU(4 * channels + dim, channels, 4, 2, dim=dim)

        self.pred0 = upconvolve(2 * channels + dim, dim, 4, 2, dim=dim)

    def forward(self, fixed, moving):
        concat_image = torch.cat((fixed, moving), dim=1)
        x1 = self.conv1(concat_image)  # 64
        x2 = self.conv2(x1)  # 32
        x3 = self.conv3(x2)  # 16
        x3_1 = self.conv3_1(x3)
        x4 = self.conv4(x3_1)  # C128, 8
        x4_1 = self.conv4_1(x4)

        pred4 = self.pred4(x4_1)
        upsamp4to3 = self.upsamp4to3(pred4)
        deconv3 = self.deconv3(x4_1)
        concat3 = torch.cat([x3_1, deconv3, upsamp4to3], dim=1)  # C(128+3), 16

        pred3 = self.pred3(concat3)
        upsamp3to2 = self.upsamp3to2(pred3)
        deconv2 = self.deconv2(concat3)
        concat2 = torch.cat([x2, deconv2, upsamp3to2], dim=1)  # C(64+3), 32

        pred2 = self.pred2(concat2)
        upsamp2to1 = self.upsamp2to1(pred2)
        deconv1 = self.deconv1(concat2)
        concat1 = torch.cat([x1, deconv1, upsamp2to1], dim=1)  # C(32+3), 64

        pred0 = self.pred0(concat1)  # 128

        return {
            "flow": pred0 * 20 * self.flow_multiplier,
        }


class RecursiveCascadeNetwork(nn.Module):
    def __init__(self, n_cascades, im_size, network, stn):
        super(RecursiveCascadeNetwork, self).__init__()
        self.det_factor = 0.1
        self.ortho_factor = 0.1
        self.reg_factor = 1.0
        self.gamma = 0.15
        self.lamb = 0.1
        self.beta = 0.1
        self.network = network
        self.n_cascades = n_cascades

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.stems = []
        # See note in base_networks.py about the assumption in the image shape
        if network == "VTNlowRes":
            print("VTNlowRes")
            self.stems.append(VTNAffineStemlowRes(im_size=im_size))
            for i in range(n_cascades):
                self.stems.append(VTNlowRes(dim=len(im_size), flow_multiplier=1.0 / n_cascades, im_size=im_size))
        else:
            raise NotImplementedError(network)

        for model in self.stems:
            model.to(device)

        self.reconstruction = stn

    def forward(self, fixed, moving):
        stem_results = []

        # Block 0
        block_result = self.stems[0](fixed, moving)  # keys: ["flow", "W", "b"]
        block_result["warped"] = self.reconstruction(moving, block_result["flow"])
        block_result["agg_flow"] = block_result["flow"]

        stem_results.append(block_result)
        # Block i
        for block in self.stems[1:]:
            block_result = block(fixed, stem_results[-1]["warped"])  # keys: ["flow"]
            if len(stem_results) == 1 and 'W' in stem_results[-1]:
                # Block 0 is Affine
                I = torch.eye(3, dtype=torch.float32).reshape(1, 3, 3).cuda()
                block_result["agg_flow"] = torch.einsum('bij,bjxyz->bixyz',
                                                        stem_results[-1]['W'] + I,
                                                        block_result['flow']
                                                        ) + stem_results[-1]['flow']
            else:
                # Block 0 is Deform or following Blocks
                block_result["agg_flow"] = self.reconstruction(stem_results[-1]["agg_flow"],
                                                               block_result["flow"]
                                                               ) + block_result["flow"]
            block_result["warped"] = self.reconstruction(moving, block_result["agg_flow"])
            stem_results.append(block_result)

        flows = []
        warps = []
        for res in stem_results:
            flows.append(res["agg_flow"])
            warps.append(res["warped"])

        return flows, warps, stem_results


class Dense3DSpatialTransformer(nn.Module):
    def __init__(self):
        super(Dense3DSpatialTransformer, self).__init__()

    def forward(self, input1, input2):
        input1 = (input1[:, :1] + 1) / 2.0
        return self._transform(input1, input2[:, 0], input2[:, 1], input2[:, 2])

    def _transform(self, input1, dDepth, dHeight, dWidth):
        batchSize = dDepth.shape[0]
        dpt = dDepth.shape[1]
        hgt = dDepth.shape[2]
        wdt = dDepth.shape[3]

        D_mesh, H_mesh, W_mesh = self._meshgrid(dpt, hgt, wdt)
        D_mesh = D_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        H_mesh = H_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        W_mesh = W_mesh.unsqueeze_(0).expand(batchSize, dpt, hgt, wdt)
        D_upmesh = dDepth + D_mesh
        H_upmesh = dHeight + H_mesh
        W_upmesh = dWidth + W_mesh

        return self._interpolate(input1, D_upmesh, H_upmesh, W_upmesh)

    def _meshgrid(self, dpt, hgt, wdt):
        d_t = torch.linspace(0.0, dpt - 1.0, dpt).unsqueeze_(1).unsqueeze_(1).expand(dpt, hgt, wdt).to('cuda:0')
        h_t = torch.matmul(torch.linspace(0.0, hgt - 1.0, hgt).unsqueeze_(1), torch.ones((1, wdt))).to('cuda:0')
        h_t = h_t.unsqueeze_(0).expand(dpt, hgt, wdt)
        w_t = torch.matmul(torch.ones((hgt, 1)), torch.linspace(0.0, wdt - 1.0, wdt).unsqueeze_(1).transpose(1, 0)).to(
            'cuda:0')
        w_t = w_t.unsqueeze_(0).expand(dpt, hgt, wdt)
        return d_t, h_t, w_t

    def _interpolate(self, input, D_upmesh, H_upmesh, W_upmesh):
        nbatch = input.shape[0]
        nch = input.shape[1]
        depth = input.shape[2]
        height = input.shape[3]
        width = input.shape[4]

        img = torch.zeros(nbatch, nch, depth + 2, height + 2, width + 2).to('cuda:0')
        img[:, :, 1:-1, 1:-1, 1:-1] = input

        imgDpt = img.shape[2]
        imgHgt = img.shape[3]  # 256+2 = 258
        imgWdt = img.shape[4]  # 256+2 = 258

        # D_upmesh, H_upmesh, W_upmesh = [D, H, W] -> [BDHW,]
        D_upmesh = D_upmesh.view(-1).float() + 1.0  # (BDHW,)
        H_upmesh = H_upmesh.view(-1).float() + 1.0  # (BDHW,)
        W_upmesh = W_upmesh.view(-1).float() + 1.0  # (BDHW,)

        # D_upmesh, H_upmesh, W_upmesh -> Clamping into [0, 257] -- index
        df = torch.floor(D_upmesh).int()
        dc = df + 1
        hf = torch.floor(H_upmesh).int()
        hc = hf + 1
        wf = torch.floor(W_upmesh).int()
        wc = wf + 1

        df = torch.clamp(df, 0, imgDpt - 1)  # (BDHW,)
        dc = torch.clamp(dc, 0, imgDpt - 1)  # (BDHW,)
        hf = torch.clamp(hf, 0, imgHgt - 1)  # (BDHW,)
        hc = torch.clamp(hc, 0, imgHgt - 1)  # (BDHW,)
        wf = torch.clamp(wf, 0, imgWdt - 1)  # (BDHW,)
        wc = torch.clamp(wc, 0, imgWdt - 1)  # (BDHW,)

        # Find batch indexes
        rep = torch.ones([depth * height * width, ]).unsqueeze_(1).transpose(1, 0).to('cuda:0')
        bDHW = torch.matmul((torch.arange(0, nbatch).float() * imgDpt * imgHgt * imgWdt).unsqueeze_(1).to('cuda:0'),
                            rep).view(-1).int()

        # Box updated indexes
        HW = imgHgt * imgWdt
        W = imgWdt
        # x: W, y: H, z: D
        idx_000 = bDHW + df * HW + hf * W + wf
        idx_100 = bDHW + dc * HW + hf * W + wf
        idx_010 = bDHW + df * HW + hc * W + wf
        idx_110 = bDHW + dc * HW + hc * W + wf
        idx_001 = bDHW + df * HW + hf * W + wc
        idx_101 = bDHW + dc * HW + hf * W + wc
        idx_011 = bDHW + df * HW + hc * W + wc
        idx_111 = bDHW + dc * HW + hc * W + wc

        # Box values
        img_flat = img.view(-1, nch).float()  # (BDHW,C) //// C=1

        val_000 = torch.index_select(img_flat, 0, idx_000.long())
        val_100 = torch.index_select(img_flat, 0, idx_100.long())
        val_010 = torch.index_select(img_flat, 0, idx_010.long())
        val_110 = torch.index_select(img_flat, 0, idx_110.long())
        val_001 = torch.index_select(img_flat, 0, idx_001.long())
        val_101 = torch.index_select(img_flat, 0, idx_101.long())
        val_011 = torch.index_select(img_flat, 0, idx_011.long())
        val_111 = torch.index_select(img_flat, 0, idx_111.long())

        dDepth = dc.float() - D_upmesh
        dHeight = hc.float() - H_upmesh
        dWidth = wc.float() - W_upmesh

        wgt_000 = (dWidth * dHeight * dDepth).unsqueeze_(1)
        wgt_100 = (dWidth * dHeight * (1 - dDepth)).unsqueeze_(1)
        wgt_010 = (dWidth * (1 - dHeight) * dDepth).unsqueeze_(1)
        wgt_110 = (dWidth * (1 - dHeight) * (1 - dDepth)).unsqueeze_(1)
        wgt_001 = ((1 - dWidth) * dHeight * dDepth).unsqueeze_(1)
        wgt_101 = ((1 - dWidth) * dHeight * (1 - dDepth)).unsqueeze_(1)
        wgt_011 = ((1 - dWidth) * (1 - dHeight) * dDepth).unsqueeze_(1)
        wgt_111 = ((1 - dWidth) * (1 - dHeight) * (1 - dDepth)).unsqueeze_(1)

        output = (val_000 * wgt_000 + val_100 * wgt_100 + val_010 * wgt_010 + val_110 * wgt_110 +
                  val_001 * wgt_001 + val_101 * wgt_101 + val_011 * wgt_011 + val_111 * wgt_111)
        output = output.view(nbatch, depth, height, width, nch).permute(0, 4, 1, 2, 3)  # B, C, D, H, W
        return output


class SpatialTransform(nn.Module):
    """
        This implementation was taken from:
        https://github.com/voxelmorph/voxelmorph/blob/master/voxelmorph/torch/layers.py
    """

    def __init__(self, size):
        super(SpatialTransform, self).__init__()
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing="ij")
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow, mode="bilinear"):
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=mode, align_corners=True)  # nearest is slower
