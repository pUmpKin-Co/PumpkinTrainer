import logging
from itertools import chain

from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import default_data_collator

from ..trainer.utils.distribute import is_distributed

logger = logging.getLogger("train")


def get_pile(
    tokenizer,
    block_size: int = 2048,
    cache_path: str = "~/pile_pythia",
):
    try:
        lm_datasets = load_from_disk(cache_path)
    except Exception:
        raw_datasets = load_dataset("JeanKaddour/minipile")
        train_datasets = raw_datasets["train"]
        column_names = train_datasets.column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        tokenized_datasets = train_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=8,
            remove_columns=column_names,
            desc="Running tokenizer on dataset",
        )

        block_size = min(block_size, tokenizer.model_max_length)

        def group_texts(examples):
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // block_size) * block_size
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        lm_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=8,
            desc=f"Grouping texts in chunks of {block_size}",
        )

        lm_datasets.save_to_disk(cache_path)

    return lm_datasets


def build_loader(
    tokenizer,
    max_seq_length: int = 2048,
    batch_size: int = 1,
    data_config: str = "~/pile_pythia",
):
    if data_config.paths == "pile":
        dataset = get_pile(tokenizer, max_seq_length, data_config.cache_path)

    if is_distributed():
        logger.info(f"Train data loaded with {len(dataset)} samples")
        sampler = None
    else:
        sampler = DistributedSampler(dataset)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=data_config.num_workers,
        sampler=sampler,
        shuffle=True if sampler is None else False,
        pin_memory=True,
        collate_fn=default_data_collator,
        drop_last=True,
    )

    return train_loader
