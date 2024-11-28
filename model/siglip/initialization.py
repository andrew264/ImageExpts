import math

import torch
from torch.nn.init import trunc_normal_, _calculate_fan_in_and_fan_out


def trunc_normal_tf_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    """Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \\leq \text{mean} \\leq b`.
    """
    with torch.no_grad():
        trunc_normal_(tensor, 0, 1.0, a, b)
        tensor.mul_(std).add_(mean)


def variance_scaling_(tensor: torch.Tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2
    else:
        denom = fan_in

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_tf_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        with torch.no_grad():
            tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        with torch.no_grad():
            tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor: torch.Tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")


def default_flax_embed_init(tensor: torch.Tensor):
    variance_scaling_(tensor, mode="fan_in", distribution="normal")