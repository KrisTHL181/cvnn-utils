import torch
import torch.nn as nn


class HilbertTransform(nn.Module):
    """
    方向感知的各向同性变换 (DAIT)

    结合各向同性希尔伯特变换和Riesz变换的优点：
    - 通过alpha控制方向敏感性
    - 在频域中进行方向加权
    - 生成信息更丰富的复数表示

    Args:
        image: 输入图像 [B, C, H, W]
        alpha: 方向敏感性参数 (0=各向同性, 1=方向敏感)
    """

    def __init__(self, alpha: float = 0):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float64))
        self.register_buffer("_cache_fy", None, persistent=False)
        self.register_buffer("_cache_fx", None, persistent=False)
        self.register_buffer("_cache_R_clamp", None, persistent=False)
        self._cache_h = None
        self._cache_w = None

    def _create_kernels(self, h: int, w: int, ref_tensor: torch.Tensor):
        fy = torch.fft.fftfreq(h, d=1.0, device=ref_tensor.device)
        fx = torch.fft.fftfreq(w, d=1.0, device=ref_tensor.device)
        FY, FX = torch.meshgrid(fy, fx, indexing="ij")

        R = torch.sqrt(FY**2 + FX**2)
        R_clamp = R.clamp(min=1e-8)

        return FY, FX, R_clamp

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b, c, h, w = image.shape
        alpha = torch.sigmoid(self.alpha)

        if h != self._cache_h or w != self._cache_w:
            FY, FX, R_clamp = self._create_kernels(h, w, image)
            self._cache_fy = FY
            self._cache_fx = FX
            self._cache_R_clamp = R_clamp
            self._cache_h, self._cache_w = h, w
        else:
            FY, FX, R_clamp = self._cache_fy, self._cache_fx, self._cache_R_clamp

        # 1. 各向同性核: -i (所有非DC频率)
        isotropic_kernel = torch.where(
            R_clamp == 0, 0.0 + 0j, -1j * torch.ones_like(R_clamp)  # type: ignore
        )

        # 2. Riesz 核
        riesz_y_kernel = torch.where(R_clamp == 0, 0.0 + 0j, -1j * FY / R_clamp)  # type: ignore
        riesz_x_kernel = torch.where(R_clamp == 0, 0.0 + 0j, -1j * FX / R_clamp)  # type: ignore

        # FFT
        freq = torch.fft.fft2(image, dim=(-2, -1))  # [b, c, h, w]

        # 各向同性响应
        iso_response = torch.fft.ifft2(
            freq * isotropic_kernel.unsqueeze(0).unsqueeze(0), dim=(-2, -1)
        ).real

        # Riesz 响应
        riesz_y = torch.fft.ifft2(
            freq * riesz_y_kernel.unsqueeze(0).unsqueeze(0), dim=(-2, -1)
        ).real
        riesz_x = torch.fft.ifft2(
            freq * riesz_x_kernel.unsqueeze(0).unsqueeze(0), dim=(-2, -1)
        ).real

        # 能量
        energy_iso = iso_response.pow(2)
        energy_riesz = riesz_x.pow(2) + riesz_y.pow(2)

        # 混合能量
        mixed_energy = (1 - alpha) * energy_iso + alpha * energy_riesz

        # 输出响应强度
        response = torch.sqrt(mixed_energy + 1e-8)  # [b, c, h, w]

        return response


@torch.no_grad()
def clip_grad_norm(parameters, max_norm=float("inf"), apply_wirtinger_scale=True):
    norms = []
    grad = torch.tensor(0.0)
    for p in parameters:
        if p.grad is not None:
            grad = p.grad
            if p.is_complex() and apply_wirtinger_scale:
                grad = (
                    grad * 0.5
                )  # 转为标准Wirtinger梯度！请注意这一点，见后文WirtingerAdamW
            # 计算 |grad|²
            param_norm_sq = (
                torch.sum(grad.real**2 + grad.imag**2).detach().clone()
                if grad.is_complex()
                else torch.sum(grad**2).detach().clone()
            )
            norms.append(param_norm_sq)

    total_norm_sq = (
        torch.sum(torch.stack(norms))
        if norms
        else torch.tensor(0.0, device=grad.device)
    )
    total_norm = total_norm_sq.sqrt().item()

    # 裁剪
    if max_norm != float("inf"):
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for p in parameters:
                if p.grad is not None:
                    p.grad.data.mul_(clip_coef)

    return total_norm
