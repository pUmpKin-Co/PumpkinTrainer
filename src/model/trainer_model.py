import logging

import torch.nn as nn

from .layers import SSMLLamaFlashAttention2

logger = logging.getLogger("train")


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logger.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_lora_param_maybe_zero_3(named_params):
    valid_keys = ["recurrence_module", "query_up_proj", "key_up_proj", "value_up_proj"]
    to_return = {k: v for k, v in named_params if any([x in k for x in valid_keys])}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


class TrainerModel(nn.Module):
    def __init__(self, config, model):
        super().__init__()
        self.config = config
        self.model = model

    def clear_cache(self):
        for name, module in self.model.named_modules():
            if hasattr(module, "clear_cache"):
                module.clear_cache()

    def set_no_grad(self):
        for name, param in self.model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, module in self.model.named_modules():
            if any(
                [
                    isinstance(module, cls)
                    for cls in [
                        SSMLLamaFlashAttention2,
                    ]
                ]
            ):
                for name, param in module.named_parameters():
                    if any(
                        [
                            key in name
                            for key in ["recurrence_module", "query_up_proj", "key_up_proj", "value_up_proj"]
                        ]
                    ):
                        param.requires_grad = True

            trainable_params = 0
            all_params = 0
            for name, param in self.named_parameters():
                num_params = param.numel()
                if num_params == 0 and hasattr(param, "ds_numel"):
                    num_params = param.ds_numel

                all_params += num_params
                if param.requires_grad:
                    trainable_params += num_params

            if logger is not None:
                logger.info(
                    f"Trainable params: {trainable_params:,d}, All params: {all_params:,d}, trainable: {100 * trainable_params/all_params:.2f}"
                )

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def custom_save_checkpoint(self, **kwargs):
        state_dict = self.state_dict()
        state_dict = get_lora_param_maybe_zero_3(state_dict.items())
        return state_dict

    def custom_load_checkpoint(self, state_dict, **kwargs):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = self.load_state_dict(state_dict, strict=False)
        return msg
