import numbers

import torch
import torch.nn as nn

from .ops import fused_recurrent_linear_attn_delta_rule


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

        self.key_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_proj = nn.Linear(hidden_size, low_rank_factor, bias=False)
        self.gating_func = nn.Linear(hidden_size, 1, bias=False)
        # self.in_norm = FusedRMSNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, prev_cache: torch.Tensor):
        # hidden_states = self.in_norm(hidden_states)
        key_states = self.key_proj(hidden_states)  # B x L x H
        key_states = torch.nn.functional.silu(key_states)
        value_states = self.value_proj(hidden_states)  # B x L x H
        beta = self.gating_func(hidden_states)
        beta = torch.sigmoid(beta).squeeze()

        all_h = fused_recurrent_linear_attn_delta_rule(key_states, value_states, beta, prev_cache)

        return all_h[:, -1]
