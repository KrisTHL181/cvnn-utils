import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings

from cvnn_utils import ComplexIsNotRsqWarning, ComplexModule
from cvnn_utils.activations import ComplexModLeakyReLU
from cvnn_utils.initialization import complex_kaiming_


class ComplexLinear(ComplexModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.complex(
                torch.randn(out_features, in_features),
                torch.randn(out_features, in_features),
            )
        )
        self.bias = torch.nn.Parameter(
            torch.complex(torch.zeros(out_features), torch.zeros(out_features))
        )
        complex_kaiming_(self.weight.real, self.weight.imag, in_features)
        complex_kaiming_(self.bias.real, self.bias.imag, in_features)

    def forward(self, x):
        return torch.matmul(x, self.weight.t()) + self.bias


class ComplexConv2d(ComplexModule):
    def __init__(
        self, in_channels, out_channels, kernel_size, padding=0, stride=1, bias=True
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.complex(
                torch.randn(out_channels, in_channels, kernel_size, kernel_size),
                torch.randn(out_channels, in_channels, kernel_size, kernel_size),
            )
        )
        self.bias = (
            nn.Parameter(
                torch.complex(torch.zeros(out_channels), torch.zeros(out_channels))
            )
            if bias
            else None
        )
        self.padding = padding
        self.stride = stride
        complex_kaiming_(self.weight.real, self.weight.imag, in_channels)
        if bias is not None:
            complex_kaiming_(self.bias.real, self.bias.imag, in_channels)  # type: ignore

    def forward(self, x):
        return F.conv2d(
            x, self.weight, self.bias, padding=self.padding, stride=self.stride
        )


class ComplexToReal(nn.Module):
    """将复数特征线性映射到实数logits（健壮实现）"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Kaiming初始化（complex -> real中的简单变体）

        nn.init.zeros_(self.bias)

    def forward(self, z):
        """自动处理任意形状的复数输入"""
        batch_size = z.size(0)
        z_flat = z.view(batch_size, -1)  # 展平空间维度

        # 实现 Re(W·z) = W_r*z_r + W_i*z_i
        return (
            torch.mm(z_flat.real, self.weight_real.t())
            + torch.mm(z_flat.imag, self.weight_imag.t())
            + self.bias
        )


class ComplexResBlock(ComplexModule):
    """复数残差块 (关键: 防止梯度消失)"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = ComplexConv2d(channels, channels, 3, padding=1)
        self.norm1 = ComplexStandardBatchNorm2d(channels)
        self.relu = ComplexModLeakyReLU(channels)
        self.conv2 = ComplexConv2d(channels, channels, 3, padding=1)
        self.norm2 = ComplexStandardBatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.norm1(self.conv1(x))
        out = self.relu(out)
        out = self.norm2(self.conv2(out))
        return out + residual


class ComplexDownsampleBlock(ComplexModule):
    """用于升维和下采样的复数块"""

    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = ComplexConv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.norm = ComplexStandardBatchNorm2d(out_channels)
        self.act = ComplexModLeakyReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ComplexCovarianceBatchNorm2d(ComplexModule):
    def __init__(self, num_features, eps=1e-4, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        # 仿射参数（可选）
        if self.affine:
            self.gamma_rr = nn.Parameter(torch.ones(num_features))
            self.gamma_ii = nn.Parameter(torch.ones(num_features))
            self.gamma_ri = nn.Parameter(torch.zeros(num_features))
            self.beta_r = nn.Parameter(torch.zeros(num_features))
            self.beta_i = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_buffer("gamma_rr", torch.ones(num_features))
            self.register_buffer("gamma_ii", torch.ones(num_features))
            self.register_buffer("gamma_ri", torch.zeros(num_features))
            self.register_buffer("beta_r", torch.zeros(num_features))
            self.register_buffer("beta_i", torch.zeros(num_features))

        # 运行统计
        self.register_buffer("running_Vrr", torch.ones(num_features))
        self.register_buffer("running_Vii", torch.ones(num_features))
        self.register_buffer("running_Vri", torch.zeros(num_features))
        self.register_buffer("running_mean_r", torch.zeros(num_features))
        self.register_buffer("running_mean_i", torch.zeros(num_features))

    def forward(self, z):
        B, C, H, W = z.shape

        # 展平空间维度
        zr = z.real.transpose(0, 1).reshape(C, -1)  # [C, B*H*W]
        zi = z.imag.transpose(0, 1).reshape(C, -1)

        # 计算 batch 统计量
        mean_r = zr.mean(dim=1)
        mean_i = zi.mean(dim=1)

        Vrr = zr.var(dim=1) + self.eps
        Vii = zi.var(dim=1) + self.eps
        Vri = (zr * zi).mean(dim=1) - mean_r * mean_i

        if self.training:
            # 更新运行统计
            self.running_mean_r = (
                1 - self.momentum
            ) * self.running_mean_r + self.momentum * mean_r
            self.running_mean_i = (
                1 - self.momentum
            ) * self.running_mean_i + self.momentum * mean_i
            self.running_Vrr = (
                1 - self.momentum
            ) * self.running_Vrr + self.momentum * Vrr
            self.running_Vii = (
                1 - self.momentum
            ) * self.running_Vii + self.momentum * Vii
            self.running_Vri = (
                1 - self.momentum
            ) * self.running_Vri + self.momentum * Vri
        else:
            mean_r = self.running_mean_r
            mean_i = self.running_mean_i
            Vrr = self.running_Vrr
            Vii = self.running_Vii
            Vri = self.running_Vri

        # 白化变换（使用协方差矩阵求逆）
        det = Vrr * Vii - Vri**2
        denom = det.sqrt()
        denom = denom.clamp(min=self.eps)

        # 逆协方差矩阵乘法（仿射变换）
        A_rr = Vii / denom
        A_ii = Vrr / denom
        A_ri = -Vri / denom

        # 白化
        xr = (z.real - mean_r.view(1, C, 1, 1)) * A_rr.view(1, C, 1, 1)
        xi = (z.imag - mean_i.view(1, C, 1, 1)) * A_ii.view(1, C, 1, 1)
        cross = (z.real - mean_r.view(1, C, 1, 1)) * A_ri.view(1, C, 1, 1)
        xr = xr + cross
        xi = xi + cross

        # 仿射变换
        if self.affine:
            out_r = (
                self.gamma_rr.view(1, C, 1, 1) * xr
                - self.gamma_ri.view(1, C, 1, 1) * xi
                + self.beta_r.view(1, C, 1, 1)
            )
            out_i = (
                self.gamma_ri.view(1, C, 1, 1) * xr
                + self.gamma_ii.view(1, C, 1, 1) * xi
                + self.beta_i.view(1, C, 1, 1)
            )
        else:
            out_r = xr
            out_i = xi

        return torch.complex(out_r, out_i)


class ComplexStandardBatchNorm2d(ComplexModule):
    def __init__(self, num_features, eps=1e-4, momentum=0.1, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features, dtype=torch.complex64))
            self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.complex64))
        else:
            self.register_buffer(
                "weight", torch.ones(num_features, dtype=torch.complex64)
            )
            self.register_buffer(
                "bias", torch.zeros(num_features, dtype=torch.complex64)
            )

        # 运行统计（复数形式）
        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.complex64)
        )
        self.register_buffer(
            "running_var", torch.ones(num_features, dtype=torch.float32)
        )

    def forward(self, z):
        if self.training:
            mean = z.mean(dim=[0, 2, 3])

            # 2. 计算模长方差（|z - μ|²，复数域自然度量）
            centered = z - mean.view(1, self.num_features, 1, 1)
            mag_sq = centered.real**2 + centered.imag**2  # |z - μ|²
            var = mag_sq.mean(dim=[0, 2, 3])  # E[|z - μ|²] - 实数标量

            # 3. 更新运行统计
            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
        else:
            mean, var = self.running_mean, self.running_var

        # 4. 复数白化（但保持相位等变性）
        std = torch.sqrt(var + self.eps)
        z_norm = (z - mean.view(1, self.num_features, 1, 1)) / std.view(
            1, self.num_features, 1, 1
        )

        # 5. 复数仿射变换（保持复线性）
        if self.affine:
            return z_norm * self.weight.view(
                1, self.num_features, 1, 1
            ) + self.bias.view(1, self.num_features, 1, 1)
        return z_norm


class ComplexAdaptiveAvgPool2d(ComplexModule):
    def __init__(self, output_size=1, allow_inconsistencies: bool = False):
        super().__init__()
        self.output_size = output_size
        self.allow_inconsistencies = allow_inconsistencies

        if output_size != 1 and not allow_inconsistencies:
            raise RuntimeError(
                "ComplexAdaptiveAvgPool2d with output_size > 1 is BLOCKED."
                "This operation has no invariant meaning on the complex domain — "
                "it treats real and imaginary parts independently (ℝ²-style), "
                "breaking the algebraic structure of ℂ. "
                "If you truly wish to commit this act (and accept the consequences), "
                "set allow_inconsistencies=True. "
                "But remember: you were warned."
            )
        elif output_size != 1 and allow_inconsistencies:
            warnings.warn(
                "ComplexAdaptiveAvgPool2d(output_size > 1) is mathematically ill-defined! "
                "You are now operating in ℝ×ℝ mode, not ℂ. "
                "Phase equivariance is broken. Complex linearity is lost. "
                "Generalization may suffer due to phase-dependent representations. "
                "Recommended: use output_size=1 or strided ComplexConv2d. "
                "You have been warned (again).",
                category=ComplexIsNotRsqWarning,
                stacklevel=2,
            )

    def forward(self, z):
        if self.output_size == 1:
            return z.mean(dim=[2, 3], keepdim=True)
        else:
            return F.adaptive_avg_pool2d(z, self.output_size)


class ComplexAvgPool2d(nn.Module):
    """
    警告：复数平均池化在数学上非良定义。

    当前实现将复数拆分为实部和虚部分别池化：
        P(z) = P(real(z)) + i * P(imag(z))

    这种操作：
        - 将破坏复线性与相位等变性
        - 对相位敏感

    建议：避免在中间层使用此操作。
    推荐替代方案：
        - 使用 `ComplexConv2d(..., stride > 1)` 实现可学习下采样
        - 末端使用全局平均：`z.mean(dim=[2,3], keepdim=True)`
    """

    def __init__(
        self, kernel_size, stride=None, padding=0, allow_inconsistencies: bool = False
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.allow_inconsistencies = allow_inconsistencies

        if not allow_inconsistencies:
            raise RuntimeError(
                "ComplexAvgPool2d is mathematically ill-defined and has been BLOCKED. "
                "It splits real/imaginary parts (ℝ²-style), breaking complex structure. "
                "Use 'ComplexConv2d(..., stride > 1)' for downsampling instead. "
                "If you really know what you are doing (and accept the consequences), "
                "set allow_inconsistencies=True — but you have been warned."
            )
        else:
            # 允许不一致？
            warnings.warn(
                "ComplexAvgPool2d is mathematically ill-defined! "
                "You are treating ℂ as ℝ² by pooling real and imaginary parts separately. "
                "This breaks phase equivariance and complex linearity. "
                "Only use this if you fully understand the implications (e.g., debugging). "
                "Recommended alternative: strided ComplexConv2d.",
                category=ComplexIsNotRsqWarning,
                stacklevel=2,
            )

    def forward(self, z):
        return torch.complex(
            F.avg_pool2d(z.real, self.kernel_size, self.stride, self.padding),
            F.avg_pool2d(z.imag, self.kernel_size, self.stride, self.padding),
        )


class ComplexDropout(ComplexModule):
    def __init__(self, p=0.5, mode="element"):
        super().__init__()
        if mode not in ["element", "channel", "spatial"]:
            raise ValueError("Mode must be 'element', 'channel', or 'spatial'")
        self.p = p
        self.mode = mode

    def forward(self, z):
        if not self.training:
            return z

        if self.mode == "element":
            mask = (torch.rand_like(z.real) > self.p).to(z.dtype)
            return z * mask / (1 - self.p)

        elif self.mode == "channel":
            mask = (
                torch.rand(z.shape[0], z.shape[1], 1, 1, device=z.device) > self.p
            ).to(z.dtype)
            return z * mask / (1 - self.p)

        elif self.mode == "spatial":
            mask = (
                torch.rand(z.shape[0], 1, *z.shape[2:], device=z.device) > self.p
            ).to(z.dtype)
            return z * mask / (1 - self.p)

    def extra_repr(self):
        return f"p={self.p}, mode={self.mode}"
