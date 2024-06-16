import logging
from collections import defaultdict

import torch
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
        for name, module in self.named_modules():
            if hasattr(module, "clear_cache"):
                module.clear_cache()

    def set_no_grad(self):
        for name, param in self.named_parameters():
            if "lora" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        for name, module in self.named_modules():
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
                    print(name)
                    trainable_params += num_params

            if logger is not None:
                logger.info(
                    f"Trainable params: {trainable_params:,d}, All params: {all_params:,d}, trainable: {100 * trainable_params/all_params:.2f}"
                )

    def forward(self, batch):
        input_ids = torch.tensor_split(
            batch["input_ids"],
            list(range(self.config.chunk_size, batch["input_ids"].shape[1], self.config.chunk_size)),
            dim=1,
        )
        if "attention_mask" in batch:
            attention_mask = torch.tensor_split(
                batch["attention_mask"],
                list(range(self.config.chunk_size, batch["attention_mask"].shape[1], self.config.chunk_size)),
                dim=1,
            )

        if "labels" in batch:
            labels = torch.tensor_split(
                batch["labels"],
                list(range(self.config.chunk_size, batch["labels"].shape[1], self.config.chunk_size)),
                dim=1,
            )

        self.clear_cache()
        past_statistic = defaultdict(list)
        for i in range(len(input_ids) - 1):
            outputs = self.model(
                input_ids=input_ids[i],
                attention_mask=attention_mask[i] if "attention_mask" in batch else None,
                labels=labels[i] if "labels" in batch else None,
                output_hidden_states=False,
                use_cache=False,
                output_attentions=False,
                past_statistic=past_statistic,
                should_build=True,
            )

            ntp_loss = outputs.loss

            self.backward(ntp_loss)

        last_outputs = self.model(
            input_ids=input_ids[-1],
            attention_mask=attention_mask[-1] if "attention_mask" in batch else None,
            labels=labels[-1] if "labels" in batch else None,
            output_hidden_states=False,
            use_cache=False,
            output_attentions=False,
            past_statistic=past_statistic,
            should_build=True,
        )
        ntp_loss = last_outputs.loss

        return ntp_loss

    def custom_save_checkpoint(self, **kwargs):
        state_dict = self.state_dict()
        state_dict = get_lora_param_maybe_zero_3(state_dict.items())
        return state_dict

    def custom_load_checkpoint(self, state_dict, **kwargs):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = self.load_state_dict(state_dict, strict=False)
        return msg
