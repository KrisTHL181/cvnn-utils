import torch.nn as nn


def complex_kaiming_(weight_real, weight_imag, in_features, out_features):
    std = (2.0 / (in_features + out_features)) ** 0.5  # fan-in + fan-out
    nn.init.normal_(weight_real, 0, std)
    nn.init.normal_(weight_imag, 0, std)
