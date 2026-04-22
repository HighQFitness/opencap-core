# Minimal substitutes for timm.models.layers (drop_path, to_2tuple, trunc_normal_)
# so ViTMoE works on Python 3.7 / old PyTorch without installing timm (avoids safetensors
# and other deps that break on stanfordnmbl/mmpose:0.1).
#
# drop_path / trunc_normal_ logic follows the timm / PyTorch MIT-style patterns.

from __future__ import annotations

import math
import warnings

import torch


def to_2tuple(x):
    if isinstance(x, tuple):
        return x
    return (x, x)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Match timm drop_path (stochastic depth)."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        if (mean < a - 2 * std) or (mean > b + 2 * std):
            warnings.warn(
                'mean is more than 2 std from [a, b] in trunc_normal_. '
                'The distribution of values may be incorrect.',
                stacklevel=2,
            )
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
