import math
import torch.nn as nn
from torch.nn.init import calculate_gain


def complex_kaiming_(
    weight_real, weight_imag, mode="fan_in", nonlinearity="relu", a=0.01
):
    """
    Complex Kaiming initialization for complex-valued weights.

    Args:
        weight_real (Tensor): Real part of weight
        weight_imag (Tensor): Imaginary part of weight
        mode (str): 'fan_in' (default) or 'fan_out'
        nonlinearity (str): Nonlinearity used after layer (default: 'relu')
        a (float): Negative slope for leaky_relu (if used)
    """
    # Compute fan-in and fan-out
    if weight_real.ndim == 2:  # Linear
        fan_in = weight_real.size(1)
        fan_out = weight_real.size(0)
    elif weight_real.ndim in [3, 4, 5]:  # Conv1d, Conv2d, Conv3d
        num_channels = weight_real.size(1)
        receptive_field_size = weight_real[2:].numel()
        fan_in = num_channels * receptive_field_size
        fan_out = weight_real.size(0) * receptive_field_size
    else:
        # Fallback for unusual shapes
        fan_in = weight_real.numel() / weight_real.size(0)
        fan_out = weight_real.numel() / weight_real.size(1)

    # Select fan mode
    if mode == "fan_in":
        n = fan_in
    elif mode == "fan_out":
        n = fan_out
    else:
        raise ValueError("mode must be 'fan_in' or 'fan_out'")

    # Get gain based on nonlinearity
    gain = calculate_gain(nonlinearity, a)  # type: ignore

    # Compute standard deviation for complex weights
    # Var(total) = Var(real) + Var(imag) = 2 * (std_each)^2
    # => std_each = gain / sqrt(2 * n)
    std = gain / math.sqrt(2 * n)

    # Initialize
    nn.init.normal_(weight_real, 0, std)
    nn.init.normal_(weight_imag, 0, std)

    return weight_real, weight_imag
