import logging
from itertools import chain

import torch
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
from transformers import default_data_collator

from ..trainer.utils.distribute import is_distributed
from .data_utils import PG19Dataset, PG19SlowRawDataset

logger = logging.getLogger("train")