import logging
from itertools import chain

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
from transformers import default_data_collator

from ..trainer.utils.distribute import is_distributed

logger = logging.getLogger("train")
