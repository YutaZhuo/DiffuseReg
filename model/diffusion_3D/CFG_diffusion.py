import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm.auto import tqdm
from . import loss


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) /
                n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    def repeat_noise(): return torch.randn(
        (1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    def noise(): return torch.randn(shape, device=device)

    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn,
            stn,
            bootstrap,
            channels=3,
            loss_type='l1',
            conditional=True,
            schedule_opt=None,
            loss_lambda=1,
            gamma=1,
            scaler=None,
            mean=None,
            std=None
    ):
        super().__init__()
        self.channels = channels
        self.denoise_fn = denoise_fn
        self.stn = stn
        self.bootstrap = bootstrap
        self.conditional = conditional
        self.loss_type = loss_type
        self.lambda_L = loss_lambda
        self.gamma = gamma
        self.scaler = scaler
        self.mean = mean
        self.std = std
        if schedule_opt is not None:
            pass
        if self.bootstrap is None:
            raise NotImplementedError

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean').to(device)
        else:
            raise NotImplementedError()
        self.loss_ncc = loss.crossCorrelation3D(1, kernel=(9, 9, 9), gamma=self.gamma).to(device)
        self.loss_reg = loss.gradientLoss("l2").to(device)
        self.loss_ssim = loss.SSIM3D(kernel_size=9).to(device)
        self.loss_ccef = loss.pearson_correlation()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
                             (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        """
        ???
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(
            self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, start):
        return (
                       extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                       - start
               ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        if condition_x is not None:
            with torch.no_grad():
                score = self.denoise_fn.forward_with_cond_scale(torch.cat([condition_x, x], dim=1), t, cond_scale=1.0)

            x_recon = self.predict_start_from_noise(
                x, t=t, noise=score)
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_recon

    @torch.no_grad()
    def p_sample_loop(self, x_in):
        """
        params x_in: the condition [M, F]
        """
        device = self.betas.device
        b, _, h, w, d = x_in.shape
        # max_size = torch.tensor(self.scaler, device=device).view(1, 3, 1, 1, 1)
        flow_mean = torch.tensor(self.mean, device=device).view(1, 3, 1, 1, 1)
        flow_std = torch.tensor(self.std, device=device).view(1, 3, 1, 1, 1)
        noise = torch.randn([b, 3, h, w, d], device=device)
        noises = [noise]
        x_starts = []
        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps,
                      ascii=" >"):
            noise, x_start = self.p_sample(noise, t, condition_x=x_in)
            # noises.append(noise)
            # if t % 200 == 0 or t == 1999:
            #     print(t)
            #     x_starts.append(x_start * flow_std + flow_mean)
        # the noise referring to flow needs to be un-normalized
        flow = noise * flow_std + flow_mean
        # flow = noise
        warpped = self.stn(x_in[:, :1], flow)  # x_in[:, :1] the moving img
        return warpped, flow, warpped, x_starts

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, clip_denoised=False, condition_x=condition_x)
        noise = noise_like(x.shape, device, repeat_noise) if t > 0 else 0.
        pred_noise = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_noise, x_start

    @torch.no_grad()
    def ddim_sample_loop(self, x_in):
        """
        params x_in: the condition [M, F]
        """
        device = self.betas.device
        b, _, h, w, d = x_in.shape
        # max_size = torch.tensor(self.scaler, device=device).view(1, 3, 1, 1, 1)
        flow_mean = torch.tensor(self.mean, device=device).view(1, 3, 1, 1, 1)
        flow_std = torch.tensor(self.std, device=device).view(1, 3, 1, 1, 1)
        noise = torch.randn([b, 3, h, w, d], device=device)
        noises = [noise]
        for t in tqdm(reversed(range(0, self.num_timesteps, 1)), desc='sampling loop time step',
                      total=self.num_timesteps,
                      ascii=" >"):
            noise, x_start = self.ddim_sample(noise, t, condition_x=x_in, eta=0.0)
            # noises.append(noise)
        # the noise referring to flow needs to be un-normalized
        flow = noise * flow_std + flow_mean  # \in (-ms, ms)
        # flow = noise
        warpped = self.stn(x_in[:, :1], flow)  # x_in[:, :1] the moving img
        return warpped, flow, warpped, flow

    @torch.no_grad()
    def ddim_sample(self, x, t, clip_denoised=True, repeat_noise=False, condition_x=None, eta=0.0):
        """
        clip_denoised() leads to difference
        between the scores from p_mean_variance() directly and from predict_noise_from_start()
        """
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((b,), t, device=device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x, t=batched_times, clip_denoised=False, condition_x=condition_x)

        score = self.predict_noise_from_start(x, batched_times, x_start)
        alpha_bar = extract(self.alphas_cumprod, batched_times, x.shape)
        alpha_bar_prev = extract(self.alphas_cumprod_prev, batched_times, x.shape)
        sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )
        noise = noise_like(x.shape, device, repeat_noise) if t > 0 else 0.
        model_mean = (
                x_start * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * score
        )
        pred_noise = model_mean + sigma * noise
        return pred_noise, x_start

    @torch.no_grad()
    def registration(self, x_in):
        img_mean = x_in.mean()
        img_std = x_in.std()
        x_in = (x_in - img_mean) / img_std

        print("Sample mode: DDPM")
        return self.p_sample_loop(x_in)
        # print("Sample mode: DDIM")
        # return self.ddim_sample_loop(x_in)

        # print("Sample mode: DDPM AVG")
        # times = 5
        # _, _, h, w, d = x_in.shape
        # device = self.betas.device
        # mean_warpped = torch.zeros(times, 1, h, w, d, device=device)
        # mean_flow = torch.zeros(times, 3, h, w, d,  device=device)
        # for i in range(times):
        #     warpped, flow, _, _ = self.p_sample_loop(x_in)
        #     mean_flow[i] = flow
        #     mean_warpped[i] = warpped
        # mean_warpped = torch.mean(mean_warpped, dim=0).unsqueeze(0)
        # mean_flow = torch.mean(mean_flow, dim=0).unsqueeze(0)
        # return mean_warpped, mean_flow, mean_warpped, mean_flow

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        # fix gama
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod,
                        t, x_start.shape) * noise
        )

    def p_losses(self, x_in, noise=None):
        """
        param x_in: the condition and x0(not contained here)
        """
        b, _, h, w, d = x_in["F"].shape
        device = x_in["F"].device
        # max_size = torch.tensor(self.scaler, device=device).view(1, 3, 1, 1, 1)
        flow_mean = torch.tensor(self.mean, device=device).view(1, 3, 1, 1, 1)
        flow_std = torch.tensor(self.std, device=device).view(1, 3, 1, 1, 1)
        # x_start = torch.zeros((b, 3, h, w, d), device=device)
        with torch.no_grad():
            # with x0
            self.bootstrap.eval()
            flows, warps, _ = self.bootstrap(x_in["F"], x_in["M"])
            x_start = (flows[-1] - flow_mean) / flow_std
            # without x0
            # x_start = torch.randn((b, 3, h, w, d), device=device)

        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy_fw = self.q_sample(x_start.detach(), t, noise)
        # x_noisy_fw = 0.0 * x_start + 1.0 * noise

        img_cat = torch.cat([x_in['M'], x_in['F']], dim=1)
        img_mean = img_cat.mean()
        img_std = img_cat.std()
        img_cat = (img_cat - img_mean) / img_std
        score = self.denoise_fn(torch.cat([img_cat, x_noisy_fw], dim=1), t)
        # Loss diff
        diff_loss = F.mse_loss(score, noise, reduction="mean")
        # since we use x0, then we can assume that the denoise network can give the initial x0
        # meaning we don't need to multiply the max_size to un-norm
        # flow_pred = (x_noisy_fw - score) * max_size  # \in (-ms_i, ms_i)
        flow_pred = self.predict_start_from_noise(x_noisy_fw, t, score)
        flow_pred = flow_pred * flow_std + flow_mean
        warpped = self.stn(x_in['M'], flow_pred)
        sim_loss = self.loss_ssim(warpped, x_in['F'])
        # sim_loss = self.loss_ccef(x_in['F'], warpped)
        reg_loss = self.loss_reg(flow_pred)
        total_loss = 1.0 * diff_loss + 1.0 * sim_loss + 0.1 * reg_loss
        # print(diff_loss.item(), sim_loss.item(), reg_loss.item())
        return diff_loss, sim_loss, reg_loss, total_loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
