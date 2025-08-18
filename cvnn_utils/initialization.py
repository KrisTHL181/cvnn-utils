import torch.nn as nn


def complex_kaiming_(weight_real, weight_imag, in_features, out_features):
    std = (2.0 / (in_features + out_features)) ** 0.5  # fan-in + fan-out
    # 能量守恒：实部虚部分别取一半方差 → std /= sqrt(2)
    std = std / (2**0.5)
    nn.init.normal_(weight_real, 0, std)
    nn.init.normal_(weight_imag, 0, std)
