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
        self.norm1 = ComplexBatchNorm2d(channels)
        self.relu = ComplexModLeakyReLU(channels)
        self.conv2 = ComplexConv2d(channels, channels, 3, padding=1)
        self.norm2 = ComplexBatchNorm2d(channels)

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
        self.norm = ComplexBatchNorm2d(out_channels)
        self.act = ComplexModLeakyReLU(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class ComplexBatchNorm2d(ComplexModule):
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


class WirtingerAdamW(torch.optim.Optimizer):
    """
    基于 Wirtinger 微积分的 AdamW 优化器
    更新方向：z ← z - η * ∂L/∂z*
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
        wirtinger_real_scaling=True,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            wirtinger_real_scaling=wirtinger_real_scaling,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | torch.Tensor | None:  # type: ignore
        loss = closure() if closure else None

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]
            lr = group["lr"]
            eps = group["eps"]
            apply_wirtinger_scale = group["wirtinger_real_scaling"]

            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    # 根据参数类型初始化方差
                    if p.is_complex():
                        state["exp_avg_sq"] = torch.zeros_like(grad.real)
                    else:
                        state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # 更新动量
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 统一计算梯度模长平方
                grad_sq = grad.abs().square()

                exp_avg_sq.mul_(beta2).add_(grad_sq, alpha=1 - beta2)

                # 偏差校正
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                exp_avg_corrected = exp_avg / bias_correction1
                exp_avg_sq_corrected = exp_avg_sq / bias_correction2
                denom = exp_avg_sq_corrected.sqrt() + eps

                # 对复数参数，PyTorch自动微分常返回`2 * ∂L/∂z̄`，但这对于数学而言有点反直觉。
                # 因此乘0.5以符合标准Wirtinger更新规则。通过下面的代码来检查：
                # >>> z = torch.tensor(1.0 + 2j, requires_grad=True)
                # >>> loss = z.real.pow(2) + z.imag.pow(2)
                # >>> loss.backward()
                # >>> print("grad = ", z.grad)
                # grad =  tensor(2.+4.j)
                # 2+4j非预期，我们希望这是1+2j。这意味着他们是这样计算的：grad = ∂L/∂x + i ∂L/∂y = 2 * (∂L/∂z*)
                # 请阅读文档：autograd#autograd-for-complex-numbers
                # 他们没有提到这里有2的系数，但实际代码计算证明这确实存在。
                # PyTorch似乎将复数认为成了两个实数分别求导并将其组合，但我们更希望复数优先。
                # 因此添加一个可选的0.5修正。
                if p.is_complex() and apply_wirtinger_scale:
                    exp_avg_corrected *= 0.5

                # 权重衰减 (复数安全)
                if weight_decay != 0.0:
                    p.data.mul_(1 - lr * weight_decay)

                # 应用更新
                update = exp_avg_corrected / denom
                p.data.sub_(update, alpha=lr)

        return loss


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
