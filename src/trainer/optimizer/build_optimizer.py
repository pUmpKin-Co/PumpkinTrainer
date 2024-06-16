import logging

import torch.nn as nn
from timm.optim.optim_factory import create_optimizer_v2

logger = logging.getLogger("train")


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword.split["."][-1] in name:
            isin = True
    return isin


def get_pretrain_param_groups(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []
    has_decay_name = []
    no_decay_name = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            len(param.shape) == 1
            or name.endswith(".bias")
            or (name.split(".")[-1] in skip_list)
            or check_keywords_in_name(name, skip_keywords)
        ):
            no_decay.append(param)
            no_decay_name.append(name)
        else:
            has_decay.append(param)
            has_decay_name.append(name)
    return [{"params": has_decay}, {"params": no_decay, "weight_decay": 0.0}]


def get_param_group(model: nn.Module):
    skip = {}
    skip_keywords = {}
    if hasattr(model, "no_weight_decay"):
        skip = model.no_weight_decay()
    if hasattr(model, "no_weight_decay_keywords"):
        skip_keywords = model.no_weight_decay_keywords()

    parameters = get_pretrain_param_groups(model, skip, skip_keywords)
    return parameters


def build_optimizer(
    model: nn.Module, name: str = "adamw", lr: float = 1e-3, wd: float = 1e-2, filter_bias_and_bn: bool = False
):
    parameters = get_param_group(model)
    return create_optimizer_v2(
        parameters,
        opt=name,
        lr=lr,
        weight_decay=wd,
        filter_bias_and_bn=filter_bias_and_bn,
    )
