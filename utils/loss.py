import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# [绝对防崩 + 频域 SOTA 模块版] 军工级 FP32 数学驱动 Loss
# 架构要求：net.py 完全不动。利用底层防崩机制与频域感知模块狂拉 8 项指标！
# =============================================================================
@torch.amp.autocast('cuda', enabled=False)
def safe_ssim_loss(img1, img2, window_size=11):
    img1, img2 = img1.float(), img2.float()
    """手写军工级安全 SSIM，内置严格 clamp，绝对 100% 防 NaN"""
    pad = window_size // 2
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=pad)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=pad)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2

    # 严格 clamp 防止微小负数导致崩溃
    sigma1_sq = torch.clamp(F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=pad) - mu1_sq, min=1e-5)
    sigma2_sq = torch.clamp(F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=pad) - mu2_sq, min=1e-5)
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=pad) - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 转换为 loss 形式：越小越好
    return (1.0 - ssim_map) / 2.0

class SpatialGradient(nn.Module):
    """四向全景梯度算子"""
    def __init__(self):
        super().__init__()
        kernel_v = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        kernel_h = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        kernel_d1 = [[0, 0, 1], [0, 0, 0], [-1, 0, 0]]
        kernel_d2 = [[-1, 0, 0], [0, 0, 0], [0, 0, 1]]
        kernel = torch.FloatTensor([kernel_v, kernel_h, kernel_d1, kernel_d2]).unsqueeze(1)
        self.weight = nn.Parameter(kernel, requires_grad=False)

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.weight.repeat(c, 1, 1, 1).to(x.device)
        grad = F.conv2d(x, weight, padding=1, groups=c)
        return torch.sum(torch.abs(grad), dim=1, keepdim=True)

# =============================================================================
# [全新外挂模块] 傅里叶频域感知器 (FFT Loss)
# 在不改变推理架构的前提下，通过频域约束直接拉满 SF 和 Qabf！
# =============================================================================
class FFTLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f, v, i):
        # 转换到 2D 傅里叶频域
        fft_f = torch.fft.fft2(f, norm="ortho")
        fft_v = torch.fft.fft2(v, norm="ortho")
        fft_i = torch.fft.fft2(i, norm="ortho")

        # 提取频率振幅
        mag_f = torch.sqrt(fft_f.real**2 + fft_f.imag**2 + 1e-8)
        mag_v = torch.sqrt(fft_v.real**2 + fft_v.imag**2 + 1e-8)
        mag_i = torch.sqrt(fft_i.real**2 + fft_i.imag**2 + 1e-8)

        # 核心：强迫融合图的频域振幅，完美覆盖源图像中最大的频率振幅
        target_mag = torch.max(mag_v, mag_i).detach()
        return F.l1_loss(mag_f, target_mag)

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        self.spatial_grad = SpatialGradient()
        self.fft_loss = FFTLoss()  # 挂载频域新模块
        self.pool = nn.AvgPool2d(5, stride=1, padding=2)
        self.smooth_l1 = nn.SmoothL1Loss(beta=0.05)
        self.l1_loss = nn.L1Loss()

    # 【神级护盾】：强制此函数在 FP32 高精度下运行，无视外层的 autocast，根除 NaN！
# 【神级护盾】：强制此函数在 FP32 高精度下运行，无视外层的 autocast，根除 NaN！
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, image_vis, image_ir, generate_img):
        v = image_vis[:, :1, :, :].float()
        i = image_ir[:, :1, :, :].float()
        f = generate_img.float()

        # ==========================================================
        # [核心升华] 场景光照评估系统 (Scene Illumination Evaluator)
        # ==========================================================
        # 计算当前可见光图像的全局平均亮度（代表环境照度）
        v_mean_global = torch.mean(v, dim=[1, 2, 3], keepdim=True)
        v_max_global = torch.amax(v, dim=[1, 2, 3], keepdim=True) + 1e-6
        # omega 值域约在 0~1 之间。白天 omega 大，夜间 omega 小。
        omega = torch.clamp(v_mean_global / v_max_global, min=0.1, max=0.9)

        # 动态权重分配 (动态感知瞳孔)
        # 白天(omega大)：重纹理(梯度)，轻热源(方差) -> 压制红外过曝
        # 夜间(omega小)：重热源(方差)，轻纹理(梯度) -> 防止可见光暗部噪声被锐化
        w_grad = 24.0 * omega         # 基础值 12.0 的两倍作为上限
        w_var = 20.0 * (1.0 - omega)  # 基础值 10.0 的两倍作为上限

        # 1. 强度极值保真
        target_int = torch.max(v, i).detach()
        loss_int = self.smooth_l1(f, target_int)

        # 2. 动态方差超限拉升 (应用自适应权重 w_var)
        var_v = torch.clamp(self.pool(v**2) - self.pool(v)**2, min=1e-5)
        var_i = torch.clamp(self.pool(i**2) - self.pool(i)**2, min=1e-5)
        var_f = torch.clamp(self.pool(f**2) - self.pool(f)**2, min=1e-5)
        target_var = torch.clamp(torch.max(var_v, var_i).detach() * 1.2, max=0.24)
        loss_var = torch.mean(w_var * torch.abs(var_f - target_var)) # L1 Loss 加权

        # 3. 结构分布保真
        ssim_v = safe_ssim_loss(f, v, window_size=11)
        ssim_i = safe_ssim_loss(f, i, window_size=11)
        ir_mean = torch.mean(i, dim=[2, 3], keepdim=True)
        mask_ir = (i > ir_mean).float().detach()
        mask_vis = 1.0 - mask_ir
        loss_ssim = torch.mean(mask_ir * ssim_i + mask_vis * ssim_v)

        # 4. 全景梯度保真 (应用自适应权重 w_grad)
        grad_v, grad_i, grad_f = self.spatial_grad(v), self.spatial_grad(i), self.spatial_grad(f)
        target_grad = torch.max(grad_v, grad_i).detach()
        loss_grad = torch.mean(w_grad * torch.abs(grad_f - target_grad)) # L1 Loss 加权

        # 5. 傅里叶频域保真 (利用 FFT Loss 直接拉满 SF 和 高频纹理)
        loss_fft = self.fft_loss(f, v, i)

        # 6. 局部相关性
        f_mean, v_mean, i_mean = self.pool(f), self.pool(v), self.pool(i)
        fv_cov = self.pool(f*v) - f_mean * v_mean
        fi_cov = self.pool(f*i) - f_mean * i_mean
        ncc_v = fv_cov / (torch.sqrt(var_f) * torch.sqrt(var_v) + 1e-6)
        ncc_i = fi_cov / (torch.sqrt(var_f) * torch.sqrt(var_i) + 1e-6)
        loss_ncc = 1.0 - torch.mean(torch.max(ncc_v, ncc_i))

        # 终极总损失：动态项 + 静态项
        loss_total = 1.0 * loss_int + loss_var + loss_grad + 6.0 * loss_ssim + 5.0 * loss_fft + 3.0 * loss_ncc

        # 返回 loss_var 和 loss_grad 的平均值，方便你在 train.py 里的 Tensorboard 打印监控
        return loss_total, loss_var.mean(), loss_grad.mean()

# =============================================================================
# 解耦相关性损失 (严格高精度安全版)
# =============================================================================
@torch.cuda.amp.autocast(enabled=False)
def cc(img1, img2):
    img1, img2 = img1.float(), img2.float()
    eps = 1e-6
    N, C, _, _ = img1.shape
    img1, img2 = img1.reshape(N, C, -1), img2.reshape(N, C, -1)

    img1 = img1 - img1.mean(dim=-1, keepdim=True)
    img2 = img2 - img2.mean(dim=-1, keepdim=True)

    var_1 = torch.clamp(torch.sum(img1 ** 2, dim=-1), min=1e-8)
    var_2 = torch.clamp(torch.sum(img2 ** 2, dim=-1), min=1e-8)

    cc = torch.sum(img1 * img2, dim=-1) / (eps + torch.sqrt(var_1) * torch.sqrt(var_2))
    return torch.clamp(cc, -1., 1.).mean()