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

        # 创建频域网格
        FY, FX, R_clamp = self._create_kernels(h, w, image)

        # 1. 各向同性核: -i (所有非DC频率)
        isotropic_kernel = torch.where(
            R_clamp == 0, 0.0 + 0j, -1j * torch.ones_like(R_clamp)
        )

        # 2. Riesz 核
        riesz_y_kernel = torch.where(R_clamp == 0, 0.0 + 0j, -1j * FY / R_clamp)
        riesz_x_kernel = torch.where(R_clamp == 0, 0.0 + 0j, -1j * FX / R_clamp)

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
