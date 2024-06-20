import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import LlamaMLP

from .layers_utils import DeltaRecurrentUpdate


class CustomLlamaMLP(LlamaMLP):
    def __init__(self, config, layer_id):
        super().__init__(config=config)
        self.in_recurrent_module = DeltaRecurrentUpdate(config.low_rank_factor, self.hidden_size)
        self.out_recurrent_module = DeltaRecurrentUpdate(config.low_rank_factor, self.intermediate_size)
        self.ffn_up_proj = nn.Linear(
            config.low_rank_factor * config.low_rank_factor, self.intermediate_size, bias=False
        )
        self.ffn_down_proj = nn.Linear(
            config.low_rank_factor * config.low_rank_factor, self.hidden_size, bias=False
        )
        self.layer_idx = layer_id
        self.past_in = None
        self.past_out = None

    def update_input_recurrence(self, past_in, past_out):
        past_in = self.in_recurrent_module(past_in, self.past_in)
        past_out = self.out_recurrent_module(past_out, self.past_out)

        self.past_in = past_in
        self.past_out = past_out

    def compute_forward(self, x, should_build=False):
        up_proj = self.up_proj(x)
        if self.past_in is not None:
            update_up_proj = self.ffn_up_proj(self.past_in)
            update_up_proj = nn.function.softmax(update_up_proj.mT, dim=-1)

            update_up_proj = x @ update_up_proj.mT

            up_proj = up_proj + update_up_proj

        gate_proj = self.act_fn(self.gate_proj(x))
        out = up_proj * gate_proj
        down_proj = self.down_proj(out)
        if self.past_out is not None:
            update_down_proj = self.ffn_down_proj(self.past_out)
            update_down_proj = nn.functional.softmax(update_down_proj, dim=-1)

            update_down_proj = out @ update_down_proj.mT

            down_proj = down_proj + update_down_proj

        if should_build:
            self.update_input_recurrence(x, out)

        return down_proj, x, out

    def forward(self, x, past_statistic=None, should_build=False):
        if past_statistic is not None and "ffn" in past_statistic and len(past_statistic["ffn"]) > self.layer_idx:
            past_in, past_out = past_statistic["ffn"][self.layer_idx]
            past_in, past_out = past_in.detach(), past_out.detach()
            self.update_input_recurrence(past_in, past_out)

        result, x, out = self.compute_forward(x, should_build=False)

        if past_statistic is not None:
            if len(past_statistic["ffn"]) > self.layer_idx:
                past_statistic["attn"][self.layer_idx] = (x, out)
            else:
                past_statistic["ffn"].append((x, out))
        return result
