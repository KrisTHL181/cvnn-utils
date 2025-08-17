import torch.nn as nn


def complex_kaiming_(weight_real, weight_imag, in_features):
    std = (1.0 / in_features) ** 0.5
    nn.init.normal_(weight_real, mean=0.0, std=std)
    nn.init.normal_(weight_imag, mean=0.0, std=std)
