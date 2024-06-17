import numbers

import torch
import torch.nn as nn


def manual_rms_norm(input, normalized_shape, weight, eps):
    # layer norm should always be calculated in float32
    dims = tuple(i for i in range(-1, -len(normalized_shape) - 1, -1))
    variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
    input = input * torch.rsqrt(variance + eps)

    if weight is None:
        return input

    # convert into half-precision if necessary
    if weight.dtype in [torch.float16, torch.bfloat16]:
        input = input.to(weight.dtype)

    return weight * input


class FusedRMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()

        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.zeros(*normalized_shape))
        else:
            self.register_parameter("weight", None)

    def forward(self, input_):
        weight = None if self.weight is None else self.weight + 1.0
        return manual_rms_norm(input_, self.normalized_shape, weight, self.eps)

    def extra_repr(self):
        return "{normalized_shape}, eps={eps}, " "elementwise_affine={elementwise_affine}".format(**self.__dict__)


class DeltaRecurrentUpdate(nn.Module):
    def __init__(self, low_rank_factor: int, hidden_size: int):
        super().__init__()

        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, low_rank_factor * low_rank_factor)
        # self.out_norm = FusedRMSNorm(low_rank_factor * low_rank_factor)

    def forward(self, hidden_states: torch.Tensor, prev_cache: torch.Tensor):
        key_states = self.key_proj(hidden_states)  # B x L x H
        value_states = self.value_proj(hidden_states)  # B x L x H

        if prev_cache is not None:
            value_states = value_states - torch.einsum("b l h, b h d -> b l d", key_states, prev_cache)
            new_cache = prev_cache + torch.einsum("b l h, b l d -> b h d", key_states, value_states)
        else:
            new_cache = torch.einsum("b l h, b l d -> b h d", key_states, value_states)

        new_cache = torch.nn.functional.normalize(new_cache, p=2, dim=-1)
        return new_cache
