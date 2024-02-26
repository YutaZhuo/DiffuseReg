import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time


class gradientLoss(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss, self).__init__()
        self.penalty = penalty

    def forward(self, input):
        dD = torch.abs(input[:, :, 1:, :, :] - input[:, :, :-1, :, :])
        dH = torch.abs(input[:, :, :, 1:, :] - input[:, :, :, :-1, :])
        dW = torch.abs(input[:, :, :, :, 1:] - input[:, :, :, :, :-1])
        if (self.penalty == "l2"):
            dD = dD * dD
            dH = dH * dH
            dW = dW * dW
        loss = (torch.mean(dD) + torch.mean(dH) + torch.mean(dW)) / 3.0
        return loss


class crossCorrelation3D(nn.Module):
    def __init__(self, in_ch, kernel=(9, 9, 9), gamma=1):
        super(crossCorrelation3D, self).__init__()
        self.in_ch = in_ch
        self.kernel = kernel
        self.gamma = gamma
        self.filt = (torch.ones([1, in_ch, self.kernel[0], self.kernel[1], self.kernel[2]])).to('cuda:0')

    def forward(self, input, target, flow):
        min_max = (-1, 1)
        target = (target - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]

        II = input * input
        TT = target * target
        IT = input * target

        flow = F.sigmoid(flow) ** self.gamma
        pad = (int((self.kernel[0] - 1) / 2), int((self.kernel[1] - 1) / 2), int((self.kernel[2] - 1) / 2))
        T_sum = F.conv3d(target, self.filt, stride=1, padding=pad)
        I_sum = F.conv3d(input, self.filt, stride=1, padding=pad)  # *flow
        TT_sum = F.conv3d(TT, self.filt, stride=1, padding=pad)
        II_sum = F.conv3d(II, self.filt, stride=1, padding=pad)  # *flow
        IT_sum = F.conv3d(IT, self.filt, stride=1, padding=pad)
        kernelSize = self.kernel[0] * self.kernel[1] * self.kernel[2]
        Ihat = I_sum / kernelSize
        That = T_sum / kernelSize

        cross = IT_sum - Ihat * T_sum - That * I_sum + That * Ihat * kernelSize
        T_var = TT_sum - 2 * That * T_sum + That * That * kernelSize
        I_var = II_sum - 2 * Ihat * I_sum + Ihat * Ihat * kernelSize
        cc = cross * cross * flow / (T_var * I_var + 1e-5)

        loss = -1.0 * torch.mean(cc)
        return loss


class pearson_correlation(nn.Module):
    def __init__(self):
        super(pearson_correlation, self).__init__()

    def forward(self, fixed, warped):
        flatten_fixed = torch.flatten(fixed, start_dim=1)
        flatten_warped = torch.flatten(warped, start_dim=1)

        mean1 = torch.reshape(torch.mean(flatten_fixed, dim=-1), [-1, 1])
        mean2 = torch.reshape(torch.mean(flatten_warped, dim=-1), [-1, 1])
        var1 = torch.mean(torch.square(flatten_fixed - mean1), dim=-1)
        var2 = torch.mean(torch.square(flatten_warped - mean2), dim=-1)
        cov12 = torch.mean((flatten_fixed - mean1) * (flatten_warped - mean2), dim=-1)
        pearson_r = cov12 / torch.sqrt((var1 + 1e-6) * (var2 + 1e-6))

        raw_loss = 1 - pearson_r
        raw_loss = torch.sum(raw_loss)

        return raw_loss


def make_gaussian_kernel3D(kernel_size, sigma) -> torch.Tensor:
    """
    Args:
        kernel_size: int
        sigma: float
    Function to create a Gaussian kernel.
    """

    x = torch.arange(kernel_size, dtype=torch.float32)
    y = torch.arange(kernel_size, dtype=torch.float32)
    z = torch.arange(kernel_size, dtype=torch.float32)

    x = x - (kernel_size - 1) / 2
    y = y - (kernel_size - 1) / 2
    z = z - (kernel_size - 1) / 2

    x, y, z = torch.meshgrid([x, y, z])
    grid = (x ** 2 + y ** 2 + z ** 2) / (2 * sigma ** 2)
    kernel = torch.exp(-grid)
    kernel = kernel / kernel.sum()
    return kernel


def _ssim_loss_fn3D(
        source,
        reference,
        kernel,
        eps=1e-8,
        c1=0.01 ** 2,
        c2=0.03 ** 2,
        use_padding=False,
) -> torch.Tensor:
    # ref: Algorithm section: https://en.wikipedia.org/wiki/Structural_similarity
    # ref: Alternative implementation: https://kornia.readthedocs.io/en/latest/_modules/kornia/metrics/ssim.html#ssim
    """
    Args:
        source: torch.Tensor,
        reference: torch.Tensor,
        kernel: torch.Tensor,
        eps: float=1e-8,
        c1: float=0.01 ** 2,
        c2: float=0.03 ** 2,
        use_padding: bool=False,
    """
    torch._assert(
        source.ndim == reference.ndim == 5,
        "SSIM: `source` and `reference` must be 5-dimensional tensors",
    )

    torch._assert(
        source.shape == reference.shape,
        "SSIM: `source` and `reference` must have the same shape, but got {} and {}".format(
            source.shape, reference.shape
        ),
    )

    B, C, H, W, D = source.shape
    kernel = kernel.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1, 1)
    if use_padding:
        pad_size = kernel.shape[2] // 2
        pad3d = (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size)
        source = F.pad(source, pad3d, "reflect")
        reference = F.pad(reference, pad3d, "reflect")

    # kernel: (out_channels, in_channels/groups, kH, kW, kD)
    mu1 = F.conv3d(source, kernel, groups=C)
    mu2 = F.conv3d(reference, kernel, groups=C)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1 * mu2
    mu_img1_sq = F.conv3d(source.pow(2), kernel, groups=C)
    mu_img2_sq = F.conv3d(reference.pow(2), kernel, groups=C)
    mu_img1_mu2 = F.conv3d(source * reference, kernel, groups=C)

    sigma1_sq = mu_img1_sq - mu1_sq
    sigma2_sq = mu_img2_sq - mu2_sq
    sigma12 = mu_img1_mu2 - mu1_mu2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim = numerator / (denominator + eps)

    # doing 1 - ssim because we want to maximize the ssim
    return 1 - ssim.mean()


class SSIM3D(nn.Module):
    def __init__(
            self,
            kernel_size: int = 11,
            max_val: float = 1.0,
            sigma: float = 1.5,
            eps: float = 1e-12,
            use_padding: bool = True,
    ) -> None:
        """SSIM loss function.

        Args:
            kernel_size: size of the Gaussian kernel
            max_val: constant scaling factor
            sigma: sigma of the Gaussian kernel
            eps: constant for division by zero
            use_padding: whether to pad the input tensor such that we have a score for each pixel
        """
        super().__init__()

        self.kernel_size = kernel_size
        self.max_val = max_val
        self.sigma = sigma

        gaussian_kernel = make_gaussian_kernel3D(kernel_size, sigma)
        self.register_buffer("gaussian_kernel", gaussian_kernel)

        self.c1 = (0.01 * self.max_val) ** 2
        self.c2 = (0.03 * self.max_val) ** 2

        self.use_padding = use_padding
        self.eps = eps

    def forward(self, source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        Args:
            source: source image of shape (batch_size, C, H, W)
            reference: reference image of shape (batch_size, C, H, W)

        Returns:
            SSIM loss of shape (batch_size,)
        """
        return _ssim_loss_fn3D(
            source,
            reference,
            kernel=self.gaussian_kernel,
            c1=self.c1,
            c2=self.c2,
            use_padding=self.use_padding,
            eps=self.eps,
        )


def elem_sym_polys_of_eigen_values(M):
    # M: (B, 3, 3)
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    sigma1 = torch.sum(torch.stack([M[0][0], M[1][1], M[2][2]], dim=0), dim=0)
    sigma2 = torch.sum(torch.stack([
        M[0][0] * M[1][1],
        M[1][1] * M[2][2],
        M[2][2] * M[0][0]
    ], dim=0), dim=0) - torch.sum(torch.stack([
        M[0][1] * M[1][0],
        M[1][2] * M[2][1],
        M[2][0] * M[0][2]
    ], dim=0), dim=0)
    sigma3 = torch.sum(torch.stack([
        M[0][0] * M[1][1] * M[2][2],
        M[0][1] * M[1][2] * M[2][0],
        M[0][2] * M[1][0] * M[2][1]
    ], dim=0), dim=0) - torch.sum(torch.stack([
        M[0][0] * M[1][2] * M[2][1],
        M[0][1] * M[1][0] * M[2][2],
        M[0][2] * M[1][1] * M[2][0]
    ], dim=0), dim=0)
    return sigma1, sigma2, sigma3


def det3x3(M):
    # M: (B, 3, 3)
    M = [[M[:, i, j] for j in range(3)] for i in range(3)]
    return torch.sum(torch.stack([
        M[0][0] * M[1][1] * M[2][2],
        M[0][1] * M[1][2] * M[2][0],
        M[0][2] * M[1][0] * M[2][1]
    ], dim=0), dim=0) - torch.sum(torch.stack([
        M[0][0] * M[1][2] * M[2][1],
        M[0][1] * M[1][0] * M[2][2],
        M[0][2] * M[1][1] * M[2][0]
    ], dim=0), dim=0)


def det_ortho_loss(W):
    I = torch.eye(3).reshape(1, 3, 3).cuda()
    A = W + I
    det = det3x3(A)  # det: (B)
    lossF = nn.MSELoss(reduction="sum")
    det_loss = lossF(det, torch.ones(det.shape[0]).cuda()) / 2
    eps = 1e-5
    epsI = torch.Tensor([[[eps * elem for elem in row] for row in Mat] for Mat in I]).cuda()
    C = torch.matmul(A.mT, A) + epsI

    s1, s2, s3 = elem_sym_polys_of_eigen_values(C)
    ortho_loss = s1 + (1 + eps) * (1 + eps) * s2 / s3 - 3 * 2 * (1 + eps)  # (B)
    ortho_loss = torch.sum(ortho_loss)
    return det_loss, ortho_loss


def loss_RCN(stem_results, hyper, reference_img):
    """
    Only supervise the final warped img. And regularize all middle stage flows.
    Turn on deep_sup, will also return the raw_loss of middle stage flows.
    """
    if hyper is None:
        det_f = 0.1
        ortho_f = 0.1
        reg_f = 1.0
    else:
        det_f = hyper["det"]
        ortho_f = hyper["ortho"]
        reg_f = hyper["reg"]

    device = reference_img.device
    ssim = SSIM3D(kernel_size=15, max_val=2, sigma=15 / 6).to(device)
    # ccef = pearson_correlation()
    regular = gradientLoss(penalty='l2').to(device)

    stem_len = len(stem_results)
    for i, block_result in enumerate(stem_results):
        if "W" in block_result:
            # Affine block
            block_result["det_loss"], block_result["ortho_loss"] = det_ortho_loss(block_result["W"])
            block_result["loss"] = block_result["det_loss"] * det_f \
                                   + block_result["ortho_loss"] * ortho_f
            if i == stem_len - 1:
                # if the Affine block is the final block
                warped = block_result["warped"]
                block_result["raw_loss"] = ssim(warped, reference_img)
                # block_result["raw_loss"] = ccef(reference_img, warped)
                block_result["loss"] = block_result["loss"] + block_result["raw_loss"]
        else:
            # Deform block
            if i == stem_len - 1:
                # if the current Deformable block is the final block
                warped = block_result["warped"]
                block_result["raw_loss"] = ssim(warped, reference_img)
                # block_result["raw_loss"] = ccef(reference_img, warped)
            block_result["reg_loss"] = regular(block_result["flow"]) * reg_f

    for i, block_result in enumerate(stem_results):
        block_result["loss"] = sum([block_result[k] for k in block_result if k.endswith("loss")])
    loss = sum([r["loss"] for r in stem_results])  # loss is the target to optimize
    return loss, stem_results[-1]["raw_loss"]

# if __name__ == "__main__":
#     data1 = torch.rand((4, 3, 8, 8, 8))
#     data2 = data1.clone()
#     data3 = torch.rand((4, 3, 8, 8, 8))
#     loss = SSIM3D(kernel_size=3)
#     print(loss(data1, data2))
#     print(loss(data1, data3))
