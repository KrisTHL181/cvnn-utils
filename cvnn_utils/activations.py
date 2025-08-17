import torch
import torch.nn as nn
import torch.nn.functional as F
from cvnn_utils import ComplexModule


class ComplexModLeakyReLU(ComplexModule):
    def __init__(self, channels, negative_slope=0.02, eps=1e-9, bias_init=-0.1):
        super().__init__()
        self.negative_slope = negative_slope
        self.bias = nn.Parameter(torch.full((1, channels, 1, 1), bias_init))
        self.eps = eps

    def forward(self, z):
        mag = torch.abs(z)
        if z.dim() == 2:
            bias = self.bias.view(1, -1)
        elif z.dim() == 4:
            bias = self.bias
        else:
            raise NotImplementedError(f"Unsupported dim: {z.dim()}")

        new_mag = F.leaky_relu(mag + bias, self.negative_slope)
        unit = z / (mag + 1e-8)
        return new_mag * unit


class ComplexModGELU(nn.Module):
    def __init__(self, learnable_bias=True):
        super().__init__()
        if learnable_bias:
            self.bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.bias = 0.0
        self.register_buffer("sqrt_2_over_pi", torch.sqrt(torch.tensor(2.0 / torch.pi)))

    def forward(self, z):
        bias_value = (
            self.bias.item() if isinstance(self.bias, nn.Parameter) else self.bias
        )
        if bias_value != 0.0:
            mag = torch.abs(z)
            gate = 0.5 * (
                1 + torch.erf((mag + bias_value) / torch.sqrt(torch.tensor(2.0)))
            )
            return z * gate
        else:
            r = torch.abs(z)
            inner = self.sqrt_2_over_pi * (r + bias_value + 0.044715 * r**3)  # type: ignore
            gate = 0.5 * (1 + torch.tanh(inner))
            return z * gate
