from functools import partial
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from ..trainer import CustomTrainerConfigError
from ..trainer.utils.distribute import get_rank
from .hf_model.custom_llama import CustomLlamaForCausalLM
from .layers.custom_attn import SSMLLamaFlashAttention2
from .trainer_model import TrainerModel


def set_layer_id(model):
    while hasattr(model, "module") and not hasattr(model, "layers"):
        model = model.module

    while hasattr(model, "model") and not hasattr(model, "layers"):
        model = model.model

    if hasattr(model, "gpt_neox"):
        model = model.gpt_neox

    if hasattr(model, "layers"):
        for idx, layer in enumerate(model.layers):
            for _, module in layer.named_modules():
                if hasattr(module, "layer_idx"):
                    module.layer_idx = idx


def get_by_name(name: str, use_flash_attention_2: bool = False):
    lower_name = name.lower()

    if "llama" in lower_name:
        target_layer = (".self_attn",)
        if use_flash_attention_2:
            attention = "ssm_flash_attention"
        else:
            attention = "ssm_attention"
    elif "pythia" in lower_name:
        target_layer = (".attention",)
        if use_flash_attention_2:
            attention = "ssm_flash_attention_gpt"
        else:
            attention = "ssm_attention_gpt"

    return attention, target_layer


def wrap_attention(
    model: nn.Module,
    attention: str,
    attention_config: Dict[str, Any],
    target_layer: Tuple[str],
):
    def _get_submodule(key):
        parent = model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = model.get_submodule(key)
        return parent, target, target_name

    attention_mapping = {
        "ssm_flash_attention": SSMLLamaFlashAttention2,
    }

    assert attention.lower() in attention_mapping.keys(), f"Attention {attention} not supported"
    attention = attention.lower()
    attention_class = attention_mapping[attention]
    attention_impl = partial(attention_class, **attention_config)

    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if any([layer in key for layer in target_layer]) and "proj" not in key:
            parent, target, target_name = _get_submodule(key)
            if isinstance(parent, attention_class):
                continue
            attention_module = attention_impl(layer_idx=target.layer_idx if hasattr(target, "layer_idx") else None)
            attention_module.to(device=model.device, dtype=model.dtype)
            attention_module.load_state_dict(target.state_dict(), strict=False)
            setattr(parent, target_name, attention_module)

    return model


def build(config):
    if "llama" in config.name.lower():
        model_config = CustomLlamaForCausalLM.config_class.from_pretrained(config.name)
        model_config.low_rank_factor = config.low_rank_factor
        model_config._attn_implementation = "flash_attention_2"
        model = CustomLlamaForCausalLM.from_pretrained(
            config.name,
            config=model_config,
            torch_dtype=torch.bfloat16,
            device_map=torch.device(get_rank()),
        )
    else:
        raise CustomTrainerConfigError(f"Model {config.name} not supported")

    attn_config = {
        "config": model_config,
    }
    attention, target_layer = get_by_name(config.name, config.use_flash_attention_2)
    model = wrap_attention(model, attention, attn_config, target_layer)

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.name,
        truth_remote_code=True,
        use_fast=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = TrainerModel(config, model)
    model.set_no_grad()

    return model, tokenizer
