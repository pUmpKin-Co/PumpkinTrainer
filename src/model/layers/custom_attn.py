import math
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from deepspeed.utils.logging import LoggerFactory
from transformers.cache_utils import Cache
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.models.llama.modeling_llama import (
    LlamaConfig,
    LlamaFlashAttention2,
    apply_rotary_pos_emb,
    repeat_kv,
)

from .layers_utils import DeltaRecurrentUpdate

logger = LoggerFactory.create_logger(__name__)


class SSMLLamaFlashAttention2(LlamaFlashAttention2):
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.recurrence_module = DeltaRecurrentUpdate(config.low_rank_factor, config.hidden_size)

        low_rank_factor = config.low_rank_factor
        key_value_dim = self.head_dim * self.num_key_value_heads
        self.query_up_proj = nn.Linear(low_rank_factor * low_rank_factor, self.hidden_size, bias=False)
        self.key_up_proj = nn.Linear(low_rank_factor * low_rank_factor, key_value_dim, bias=False)
        self.value_up_proj = nn.Linear(low_rank_factor * low_rank_factor, key_value_dim, bias=False)

        self.in_recurrence_cache = None

    def clear_cache(self):
        self.in_recurrence_cache = None

    def convert_4d_to_3d(self, hidden_states):
        if hidden_states.dim() == 3:
            return hidden_states
        bsz, num_heads, q_len, head_dim = hidden_states.size()
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states.reshape(bsz, q_len, -1)

    def convert_3d_to_4d(self, hidden_states):
        if hidden_states.dim() == 4:
            return hidden_states
        bsz, q_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(bsz, q_len, -1, self.head_dim)
        return hidden_states.transpose(1, 2)

    def compute_qkv(self, hidden_states):
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.in_recurrence_cache is not None:
            query_weight = self.query_up_proj(self.in_recurrence_cache)
            key_weight = self.key_up_proj(self.in_recurrence_cache)
            value_weight = self.value_up_proj(self.in_recurrence_cache)

            query_weight = torch.nn.functional.softmax(query_weight.mT, dim=-1)
            key_weight = torch.nn.functional.softmax(key_weight.mT, dim=-1)
            value_weight = torch.nn.functional.softmax(value_weight.mT, dim=-1)

            update_query = hidden_states @ query_weight.mT
            update_key = hidden_states @ key_weight.mT
            update_value = hidden_states @ value_weight.mT

            query_states = query_states + update_query
            key_states = key_states + update_key
            value_states = value_states + update_value

        query_states = self.convert_3d_to_4d(query_states)
        key_states = self.convert_3d_to_4d(key_states)
        value_states = self.convert_3d_to_4d(value_states)

        return query_states, key_states, value_states

    def compute_output(self, x):
        # TODO: Should we add recurrent for output?
        output = self.o_proj(x)
        return output

    def update_input_recurrence(self, hidden_states):
        if self.in_recurrence_cache is not None:
            self.in_recurrence_cache = self.in_recurrence_cache.detach()

        in_recurrence_cache = self.recurrence_module(hidden_states, self.in_recurrence_cache)
        self.in_recurrence_cache = in_recurrence_cache

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        past_statistic: Optional[Dict] = None,
        should_build: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if (
            past_statistic is not None
            and "attn" in past_statistic
            and len(past_statistic["attn"]) > self.layer_idx
        ):
            past_input = past_statistic["attn"][self.layer_idx]
            past_input = past_input.detach()
            self.update_input_recurrence(past_input)

        query_states, key_states, value_states = self.compute_qkv(hidden_states)

        if not self.training and should_build:
            self.update_input_recurrence(hidden_states)

        if past_statistic is not None:
            if len(past_statistic["attn"]) > self.layer_idx:
                past_statistic["attn"][self.layer_idx] = hidden_states
            else:
                past_statistic["attn"].append(hidden_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        dropout_rate = self.attention_dropout if self.training else 0.0
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        if query_states.dtype != key_states.dtype:
            key_states = key_states.to(query_states.dtype)
            value_states = value_states.to(query_states.dtype)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.compute_output(attn_output)

        if not output_attentions:
            attn_weights = None

        return (
            attn_output,
            attn_weights,
            past_key_value,
            (self.in_recurrence_cache,),
        )
