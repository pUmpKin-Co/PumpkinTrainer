from dataclasses import dataclass

from src.trainer.utils import ModelConfig


@dataclass
class TransformerConfig350M(ModelConfig):
    hidden_size: int = 1024
    vocab_size: int = 32000
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 1
    norm_eps: float = 1e-6
    num_heads: int = 8
    num_hidden_layers: int = 12
    num_kv_heads: int = None
    max_position_embeddings: int = 2048
    window_size: int = None
    hidden_ratio: int = 4
    use_moe: bool = True
    num_experts: int = 64
    n_shard_experts: int = None
    activate_expert: int = 8
    moe_intermediate_size: int = 1024
    hidden_act: str = "silu"
    scoring_func: str = "softmax"
    norm_topk_prob: bool = False
    gating_dim: int = 1024
    alpha: float = 0.01
