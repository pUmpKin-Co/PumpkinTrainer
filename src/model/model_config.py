from dataclasses import dataclass
from typing import Optional

from src.trainer.utils import ModelConfig
from transformers import PretrainedConfig


@dataclass
class TransformerConfig350M(ModelConfig, PretrainedConfig):
    hidden_size: int = 1024
    vocab_size: int = 32000
    norm_eps: float = 1e-6
    num_heads: int = 8
    num_hidden_layers: int = 12
    num_kv_heads: Optional[int] = None
    max_position_embeddings: int = 2048
    window_size: Optional[int] = None
    hidden_ratio: int = 4
    use_moe: bool = True
    num_experts: int = 64
    n_shard_experts: Optional[int] = None
    activate_expert: int = 8
    moe_intermediate_size: int = 1024
    hidden_act: str = "silu"
    scoring_func: str = "softmax"
    norm_topk_prob: bool = False
    gating_dim: int = 1024
    aux_loss_alpha: float = 0.01
    seq_aux: bool = False
    pruned_heads: Optional[dict] = None
    initializer_range: float = 0.02
    torchscript: bool = False

    def __init__(self, **kwargs):
        PretrainedConfig.__init__(
            self,
            tie_word_embeddings=False,
            **kwargs,  # Any additional keyword arguments will be passed
        )
