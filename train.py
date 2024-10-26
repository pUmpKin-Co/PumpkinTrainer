import logging
import sys
from dataclasses import dataclass, field
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Union

import torch
import wandb
from datasets import load_dataset, load_from_disk
from packaging import version
from src.model.model_config import TransformerConfig350M
from src.model.transformer import TransformerForCausalLM
from src.trainer.EpochBasedTrainer import EpochBasedTrainer
from src.trainer.hook.eval_hook import EpochEvalHook, IterEvalHook
from src.trainer.IterBasedTrainer import IterBasedTrainer
from src.trainer.optimizer import build_optimizer
from src.trainer.utils import (
    CustomTrainerConfigError,
    DataConfig,
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
from transformers import AutoTokenizer, default_data_collator

logger = logging.getLogger("train")


@dataclass
class CustomDataConfig(DataConfig):
    tokenizer: str = "mistralai/Mistral-7B-Instruct-v0.2"
    data: str = "cerebras/SlimPajama-627B"
    cache_path: str = "~/cache_data"
    training_size: Union[int, str] = "100B"
    validation_size: Union[int, str] = "10B"


@dataclass
class CustomTrainConfig(TrainConfig):
    model: TransformerConfig350M = field(default_factory=TransformerConfig350M)
    data: CustomDataConfig = field(default_factory=CustomDataConfig)


def main(config: TrainConfig):
    logger.info(f"Creating model")

    tokenizer = AutoTokenizer.from_pretrained(config.data.tokenizer)
    model = TransformerForCausalLM(config.model)

    # Load data
    def tokenizer_fn(examples):
        return tokenizer(examples["text"])

    def group_fn(examples, block_size):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    root_path = Path(config.data.cache_path)
    if not Path(config.data.cache_path).exists():
        logger.info(f"Loading dataset from {config.data.data}")
        dataset = load_dataset(
            config.data.data,
            trust_remote_code=True,
            split="train",
        )

        dataset = dataset.map(
            tokenizer_fn, batched=True, remove_columns=["text"], num_proc=8, desc="Tokenizing dataset"
        )

        dataset = dataset.map(
            partial(group_fn, block_size=config.model.max_position_embeddings),
            batched=True,
            num_proc=8,
            desc="Grouping texts",
        )

        val_dataset = load_dataset(
            config.data.data,
            trust_remote_code=True,
            split="validation",
        )

        val_dataset = val_dataset.map(
            tokenizer_fn, batched=True, remove_columns=["text"], num_proc=8, desc="Tokenizing dataset"
        )

        val_dataset = val_dataset.map(
            partial(group_fn, block_size=config.model.max_position_embeddings),
            batched=True,
            num_proc=8,
            desc="Grouping texts",
        )

        train_path = root_path / "train"
        train_path.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(train_path)

        val_path = root_path / "validation"
        val_path.mkdir(exist_ok=True, parents=True)
        val_dataset.save_to_disk(val_path)
    else:
        logger.info(f"Loading dataset from {config.data.cache_path}")
        dataset = load_from_disk(config.data.cache_path)
        val_dataset = load_from_disk(config.data.cache_path)

    exit(1)
    training_size = config.data.training_size
    validation_size = config.data.validation_size

    if isinstance(training_size, str):
        digits = training_size[:-1]
        unit = training_size[-1]

        assert unit in ["B", "M", "T"], f"Invalid unit {unit}"
        training_size = int(digits)
        if unit == "M":
            training_size *= 1e6
        elif unit == "T":
            training_size *= 1e12
        elif unit == "B":
            training_size *= 1e9
    else:
        training_size = int(training_size)

    select_size = training_size // config.model.max_position_embeddings
    train_dataset = dataset.select(range(select_size))

    if isinstance(validation_size, str):
        digits = validation_size[:-1]
        unit = validation_size[-1]

        assert unit in ["B", "M", "T"], f"Invalid unit {unit}"
        validation_size = int(digits)
        if unit == "M":
            validation_size *= 1e6
        elif unit == "T":
            validation_size *= 1e12
        elif unit == "B":
            validation_size *= 1e9
    else:
        validation_size = int(validation_size)

    select_size = validation_size // config.model.max_position_embeddings
    eval_dataset = val_dataset.select(range(select_size))

    logger.info(f"Creating dataloader")

    if not config.is_distribute:
        sampler = None
    else:
        sampler = torch.utils.data.DistributedSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.device_train_batch_size,
        shuffle=True if sampler is None else False,
        sampler=sampler,
        pin_memory=True,
        collate_fn=default_data_collator,
    )

    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=config.device_eval_batch_size,
        shuffle=False,
        sampler=None,
        pin_memory=True,
        collate_fn=default_data_collator,
    )

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
        "data_loader": train_loader,
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
        "eval_data_loader": eval_loader,
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
