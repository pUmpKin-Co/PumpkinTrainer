import math
import numbers

import torch
import torch.nn as nn

from .ops import pscan


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


class MambaSelectiveScan(torch.nn.Module):
    dt_max = 0.1
    dt_min = 0.001
    dt_init_floor = 1e-4
    rms_norm_eps = 1e-5
    dt_scale = 1.0

    def __init__(
        self,
        low_rank_factor: int,
        hidden_size: int,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # self.hidden_size = config.hidden_size
        self.low_rank_factor = low_rank_factor
        self.time_step_rank = self.hidden_size // self.low_rank_factor

        self.input_proj = torch.nn.Linear(self.hidden_size, self.low_rank_factor + self.time_step_rank, bias=False)
        self.dt_proj = torch.nn.Linear(self.time_step_rank, self.hidden_size, bias=True)

        A = torch.arange(1, self.low_rank_factor + 1, dtype=torch.float32).repeat(self.hidden_size, 1)
        self.A_log = torch.nn.Parameter(torch.log(A))

        self.dt_layernorm = FusedRMSNorm(self.time_step_rank, eps=self.rms_norm_eps)
        self.B_layernorm = FusedRMSNorm(self.low_rank_factor, eps=self.rms_norm_eps)

        with torch.no_grad():
            dt_init_std = self.time_step_rank**-0.5
            self.dt_proj.weight.uniform_(-dt_init_std, dt_init_std)

            dt = torch.exp(
                torch.rand(self.hidden_size) * (math.log(self.dt_max) - math.log(self.dt_min))
                + math.log(self.dt_min)
            ).clamp(min=self.dt_init_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                self.dt_proj.bias.copy_(inv_dt)

    def apply_layernorm(self, B, delta):
        B = self.B_layernorm(B)
        delta = self.dt_layernorm(delta)

        return B, delta

    def forward(self, hidden_states: torch.Tensor, initial_state: torch.Tensor = None):
        if initial_state is not None:
            initial_state = initial_state.detach()

        A = -torch.exp(self.A_log.float())
        delta_B = self.input_proj(hidden_states)
        B, delta = delta_B.split([self.low_rank_factor, self.time_step_rank], dim=-1)
        B, delta = self.apply_layernorm(B, delta)

        delta = self.dt_proj(delta)
        delta = torch.nn.functional.softplus(delta)

        states = self.run_scan(hidden_states, delta, A, B, initial_state)
        return states[:, -1]

    def run_scan(self, hidden_states, delta, A, B, initial_state):
        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(-2)
        gating_input = deltaB * (hidden_states.unsqueeze(-1))

        if initial_state is not None:
            B, L, D, N = deltaA.shape
            deltaA = torch.cat([deltaA.new_zeros(B, 1, D, N), deltaA], dim=1)
            gating_input = torch.cat([initial_state.unsqueeze(1), gating_input], dim=1)

        states = pscan(deltaA, gating_input)
        return states
