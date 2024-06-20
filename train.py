import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import wandb
from packaging import version
from src.data.build_data import build_loader, build_pg_loader
from src.model.build_model import build
from src.trainer.EpochBasedTrainer import EpochBasedTrainer
from src.trainer.hook.eval_hook import EpochEvalHook, IterEvalHook
from src.trainer.IterBasedTrainer import IterBasedTrainer
from src.trainer.optimizer import build_optimizer
from src.trainer.utils import (
    CustomTrainerConfigError,
    DataConfig,
    ModelConfig,
    TrainConfig,
    barrier,
    deepspeed_init_distributed,
    get_default_device,
    get_fsdp_wrap_policy,
    init_distributed,
    seed_all,
    setup_logger,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

logger = logging.getLogger("train")


@dataclass
class CustomModelConfig(ModelConfig):
    name: str = "gpt2"
    max_seq_length: int = 2048
    chunk_size: int = 2048
    use_flash_attention_2: bool = True
    low_rank_factor: int = 16


@dataclass
class CustomDataConfig(DataConfig):
    cache_path: str = "~/pile_tinyllama"


@dataclass
class CustomTrainConfig(TrainConfig):
    model: CustomModelConfig = field(default_factory=CustomModelConfig)
    data: CustomDataConfig = field(default_factory=CustomDataConfig)


def main(config: TrainConfig):
    logger.info(f"Creating model")
    model, tokenizer = build(config.model)

    logger.info(f"Creating dataloader")
    dataloader = build_loader(
        tokenizer,
        batch_size=config.device_train_batch_size,
        max_seq_length=config.model.max_seq_length,
        data_config=config.data,
    )

    logger.info(f"Creating Evaluation dataloader")
    eval_dataloader = build_pg_loader(
        tokenizer,
        chunk_size=config.model.chunk_size,
        file_path=config.evaluators.data.paths,
    )

    if hasattr(model, "gradient_checkpointing_enable") and config.activation_checkpointing:
        model.model.gradient_checkpointing_enable()
        model.model.enalbe_input_requre_grads()

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
        import deepspeed

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

        model, optimizer, _, _ = deepspeed.initialize(
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
        "work_dir": config.save_folder,
        "max_num_checkpoints": config.checkpoint.save_num_checkpoints_to_keep,
        "log_period": config.console_log_interval,
        "ckpt_period": config.save_interval,
        "clip_grad_norm": config.max_grad_norm,
        "enable_amp": config.autocast_precision != torch.float32,
        "accelerator": config.accelerator,
        "cumulative_iters": config.gradient_accumulation_steps,
        "eval_data_loader": None,
        "is_distributed": config.is_distribute,
        "deepspeed": config.deepspeed.enabled,
        "fsdp": config.fsdp.enabled,
        "torch_compile": config.compile,
        "dtype": config.autocast_precision,
        "save_ckpt_by": config.checkpoint.save_strategy,
        "eval_data_loader": eval_dataloader,
    }

    if config.run_strategy == "step":
        trainer = IterBasedTrainer(max_iters=config.run_duration, **share_args)
    else:
        trainer = EpochBasedTrainer(max_epochs=config.run_duration, **share_args)

    if config.evaluators is not None:
        if config.eval_interval > config.run_duration:
            trainer.register_hook([IterEvalHook(evaluators=config.evaluators.type, period=config.eval_interval)])
        else:
            trainer.register_hook([EpochEvalHook(evaluators=config.evaluators.type, period=config.eval_interval)])

    if config.load_path is not None:
        resume_path = config.load_path
    else:
        resume_path = None

    trainer.train(load_checkpoint=resume_path)

    final_ckpt_dir = Path(trainer.ckpt_dir)
    ckpt = model.custom_save_checkpoint()
    torch.save(ckpt, final_ckpt_dir / "final_checkpoint.pt")


if __name__ == "__main__":
    try:
        if "--local_rank" in sys.argv[1]:
            config_path, other_args = sys.argv[2], sys.argv[3:]
        else:
            config_path, other_args = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise CustomTrainerConfigError(f"Usage: [--local_rank] {sys.argv[0]} CONFIG_PATH [OTHER_ARGS]")

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
            raise CustomTrainerConfigError(f"{save_path} already exists, use save_overwrite=true to overwrite")
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
