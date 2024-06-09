import logging
import sys
from pathlib import Path

import torch
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from dataclasses import dataclass, field
from Customtrainer.EpochBasedTrainer import EpochBasedTrainer
from Customtrainer.IterBasedTrainer import IterBasedTrainer
from Customtrainer.optimizer import build_optimizer
from Customtrainer.utils import (
    CustomTrainerConfigError,
    TrainConfig,
    ModelConfig,
    deepspeed_init_distributed,
    init_distributed,
    setup_logger,
    seed_all,
    barrier,
    get_fsdp_wrap_policy,
    get_default_device,
)

logger = logging.getLogger("train")


@dataclass
class CustomModelConfig(ModelConfig):
    name: str = "gpt2"


@dataclass
class CustomTrainConfig(TrainConfig):
    model: CustomModelConfig = field(default_factory=CustomModelConfig)


def main(config: TrainConfig):
    logger.info(f"Creating dataloader")
    """
    *****************************************
    *                                       *
    *                                       *
    * Building the dataloader               *
    * dataloader = build_loader(config)     *
    *                                       *
    ***************************************
    """
    dataloader = None

    logger.info(f"Creating model")
    """
    *****************************************
    *                                       *
    *                                       *
    * Building the model                    *
    * model = build_model(config.model)     *
    *                                       *
    *****************************************
    """
    model = None
    if hasattr(model, "num_params"):
        logger.info(f"Model has {model.num_params()} parameters")

    if hasattr(model, "num_params"):
        logger.info(f"Model has {model.num_params()} parameters")

    if hasattr(model, "gradient_checkpointing_enable") and config.activation_checkpointing:
        model.gradient_checkpointing_enable()
        model.enalbe_input_requre_grads()

    if config.fsdp.enabled:
        if hasattr(model, "get_fsdp_wrap_policy"):
            wrap_policy = model.get_fsdp_wrap_policy()
        elif hasattr(model, "block"):
            wrap_policy = get_fsdp_wrap_policy(type(model.block))
        else:
            wrap_policy = True

        torch.cuda.set_device(f"cuda:{config.local_rank}")
        device = torch.device("cuda")

        if version.parse(torch.__version__) >= version.parse("2.1.0"):
            # This prevents any parameters from being initialized twice
            def dummy_init_fn(module: torch.nn.Module) -> None:
                module.to_empty(device=get_default_device())

            param_init_fn = dummy_init_fn
        else:
            param_init_fn = None

        model = FSDP(
            model,
            sharding_strategy=config.fsdp.sharding_strategy,
            mixed_precision=config.fsdp_precision,
            auto_wrap_policy=wrap_policy,
            use_orig_params=config.fsdp.use_orig_params,
            limit_all_gathers=True,
            device_id=config.local_rank,
            param_init_fn=param_init_fn,
        )

        optimizer = build_optimizer(
            model,
            name=config.optimizer.name,
            lr=config.optimizer.learning_rate,
            wd=config.optimizer.weight_decay,
            filter_bias_and_bn=config.optimizer.decay_norm_and_bias,
        )

    elif config.deepspeed.enabled:
        from deepspeed import deepspeed

        if config.optimizer.name == "adamw":
            parameter = None
            optimizer = None
        else:
            parameter = None
            optimizer = build_optimizer(
                model,
                name=config.optimizer.name,
                lr=config.optimizer.learning_rate,
                wd=config.optimizer.weight_decay,
                filter_bias_and_bn=config.optimizer.decay_norm_and_bias,
            )

        model, optimizer, _ = deepspeed.initialize(
            model=model,
            config=config.deepspeed_init,
            optimizer=optimizer if optimizer is not None else None,
            model_parameters=parameter if parameter is not None else None,
        )
    else:
        optimizer = build_optimizer(
            model,
            name=config.optimizer.name,
            lr=config.optimizer.learning_rate,
            wd=config.optimizer.weight_decay,
            filter_bias_and_bn=config.optimizer.decay_norm_and_bias,
        )

    share_args = {
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": config.scheduler,
        "data_loader": dataloader,
        "word_dir": config.save_folder,
        "max_num_checkpoints": config.checkpoint.save_num_checkpoints_to_keep,
        "log_period": config.console_log_interval,
        "ckpt_period": config.save_interval,
        "clip_grad_norm": config.max_grad_norm,
        "enable_amp": config.autocast_precision != torch.float32,
        "accelerator": config.accelerator,
        "cumulative_iters": config.gradient_accumulation_steps,
        "eval_data_loader": None,
        "is_distributed": config.is_distribute,
        "deep_speed": config.deepspeed.enabled,
        "fsdp": config.fsdp.enabled,
        "torch_compile": config.compile,
        "dtype": config.autocast_precision,
        "save_ckpt_by": config.checkpoint.save_strategy,
    }

    if config.run_strategy.step:
        trainer = IterBasedTrainer(max_iters=config.run_duration, **share_args)
    else:
        trainer = EpochBasedTrainer(max_epochs=config.run_duration, **share_args)

    if config.load_path is not None:
        resume_path = config.load_path
    else:
        resume_path = None

    trainer.train(load_checkpoint=resume_path)


if __name__ == "__main__":
    try:
        config_path, other_args = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise CustomTrainerConfigError(f"Usage: {sys.argv[0]} CONFIG_PATH [OTHER_ARGS]")

    config = CustomTrainConfig.load(config_path, other_args)

    if config.deepspeed.enabled:
        config.rank, config.local_rank, config.world_size = deepspeed_init_distributed()
        config.is_distribute = config.world_size > 1
    else:
        config.rank, config.local_rank, config.world_size = init_distributed()
        config.is_distribute = config.world_size > 1

    setup_logger("train", output=config.save_folder, rank=config.rank)
    seed_all(config.seed)

    if config.rank == 0:
        save_path = Path(config.save_folder) / "config.yaml"
        if save_path.is_file() and not config.save_overwrite:
            raise CustomTrainerConfigError(f"{save_path} already exists, use --save_overwrite to overwrite")
        else:
            logger.info(f"Saving config to {save_path}")
            save_path.parent.mkdir(exist_ok=True, parents=True)
            config.save(save_path)
        del save_path

    barrier()

    if config.wandb is not None and config.wandb.enabled and (config.rank == 0 or not config.wandb.rank_zero_only):
        wandb_dir = Path(config.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            dir=wandb_dir,
            project=config.wandb.project,
            entity=config.wandb.entity,
            group=config.wandb.group,
            name=config.wandb.name,
            tags=config.wandb.tags,
            config=config.asdict(exclude=["wandb"]),
        )

    barrier()
    main(config)
