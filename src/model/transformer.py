import logging
import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel

from .activations import swiglu_linear
from .model_config import TransformerConfig
from .utils import FlashAttention, RMSNorm

logger = logging.get_logger(__name__)


class SwiGLUMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        expand_ratio: int = 4,
        intermediate_size: Optional[int] = None,
        hidden_act: Optional[str] = "gelu",
    ):
        super(SwiGLUMLP, self).__init__()

        self.intermediate_size = intermediate_size or hidden_size * expand_ratio
        self.gate_proj = nn.Linear(hidden_size, self.intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        y = self.gate_proj(x)
        gate, proj = y.chunk(2, dim=-1)
        return swiglu_linear(gate, proj, self.down_proj.weight, self.down_proj.bias)


class MoEGate(nn.Module):
    def __init__(self, config):
        self.config = config
        self.top_k = config.activate_expert
        self.num_experts = config.num_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        bsz, seq_len, h = x.shape

        hidden_states = x.view(bsz * seq_len, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == "softmax":
            scores = logits.softmax(dim=-1)
        else:
            raise ValueError(f"Unknown scoring function: {self.scoring_func}")

        topk_weight, topk_idx = torch.topk(scores, self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.alpha > 0.0 and self.training:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_axu_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.num_experts, device=x.device)
                ce.scatter_add_(
                    1, topk_idx_for_axu_loss, torch.ones(bsz, seq_len * aux_topk, device=x.device)
                ).div_(seq_len * aux_topk / self.num_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)), sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_axu_loss.view(-1), num_classes=self.num_experts)
                ce = mask_ce.float().mean(dim=0)
                Pi = scores_for_aux.mean(dim=0)
                fi = ce * self.num_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = None

        return topk_idx, topk_weight, aux_loss


class AddAuxiliaryLoss(torch.autograd.Function):
    """
    The trick function of adding auxiliary (aux) loss,
    which includes the gradient of the aux loss during backpropagation.
    """

    @staticmethod
    def forward(ctx, x, loss):
        assert loss.numel() == 1
        ctx.dtype = loss.dtype
        ctx.required_aux_loss = loss.requires_grad
        return x

    @staticmethod
    def backward(ctx, grad_output):
        grad_loss = None
        if ctx.required_aux_loss:
            grad_loss = torch.ones(1, dtype=ctx.dtype, device=grad_output.device)
        return grad_output, grad_loss


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activate_expert = config.activate_expert
        self.experts = nn.ModuleList(
            [
                SwiGLUMLP(
                    config.hidden_size,
                    intermediate_size=config.moe_intermediate_size,
                    hidden_act=config.hidden_act,
                )
                for _ in range(config.num_experts)
            ]
        )
        self.gate = MoEGate(config)
        if config.n_shard_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shard_experts
            self.shared_experts = SwiGLUMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
            )

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        topk_idx, topk_weight, aux_loss = self.gate(x)
        hidden_states = x.view(orig_shape[0], orig_shape[1], -1)
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.activate_expert, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
            y = AddAuxiliaryLoss.apply(y, aux_loss)
        else:
            y = self.moe_infer(hidden_states, flat_topk_idx, topk_weight.view((-1, 1))).view(*orig_shape)

        if self.config.n_shard_experts is not None:
            y = y + self.shared_experts(identity)
        return y

    @torch.no_grad()
    def moe_infer(self, hidden_states, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(hidden_states)
        idx = flat_expert_indices.argsort()
        token_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idx // self.num_experts_per_token
        for i, end_idx in enumerate(token_per_expert):
            start_idx = 0 if i == 0 else token_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = hidden_states[exp_token_idx]
            expert_output = expert(expert_tokens)
            expert_output.mul_(flat_expert_weights[idx[start_idx:end_idx]])
            expert_cache.scatter_reduce_(
                0, exp_token_idx.view(-1, 1).repeat(1, hidden_states.shape[-1]), expert_output, reduce="sum"
            )
        return expert_cache


class TransformerBlock(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        self.attn = FlashAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            window_size=config.window_size,
            max_position_embeddings=config.max_position_embeddings,
            layer_idx=layer_idx,
        )
        self.mlp_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.norm_eps)
        if config.use_moe:
            self.mlp = MoE(config)
        else:
            self.mlp = SwiGLUMLP(
                hidden_size=config.hidden_size,
                hidden_ratio=config.hidden_ratio,
                hidden_act=config.hidden_act,
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)
        hidden_states, attentions, past_key_values = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, residual = self.mlp_norm(hidden_states, residual, True)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attentions,)

        if use_cache:
            outputs += (past_key_values,)

        return outputs


class TransformerPreTrainedModel(PreTrainedModel):

    config_class = TransformerConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ["o_proj.weight", "down_proj.weight"]:
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class TransformerPreTrainedModel(PreTrainedModel):

    config_class = TransformerConfig
    supports_gradient_checkpointing = True
    _no_split_modules = ["TransformerBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(
        self,
        module: nn.Module,
        rescale_prenorm_residual: bool = True,
        num_residuals_per_layer: int = 2,
    ):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        if rescale_prenorm_residual:
            # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
            #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
            #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
            #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
            #
            # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
            for name, p in module.named_parameters():
                if name in ["o_proj.weight", "down_proj.weight"]:
                    # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                    # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                    # We need to reinit p since this code could be called multiple times
                    # Having just p *= scale would repeatedly scale it down
                    with torch.no_grad():
                        p /= math.sqrt(num_residuals_per_layer * self.config.num_hidden_layers)


class TransformerModel(TransformerPreTrainedModel):

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if output_attentions:
            warnings.warn(
                "`TransformerModel` does not support output attention weights now, so `output_attentions` is set to `False`."
            )
            output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)

        # embed positions
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        all_hidden_states = () if output_hidden_states else None
        all_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer.__call__, hidden_states, attention_mask, past_key_values, output_attentions, use_cache
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_attns] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attns,
        )


class TransformerForCausalLM(TransformerPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: TransformerConfig):
        super().__init__(config)

        self.model = TransformerModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor = None,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], loss_fct.ignore_index)), 1)
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
