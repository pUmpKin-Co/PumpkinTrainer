import torch
import torch.nn as nn


class DeltaRecurrentUpdate(nn.Module):
    def __init__(self, low_rank_factor: int, hidden_size: int):
        super().__init__()

        self.key_proj = nn.Linear(low_rank_factor, hidden_size)
        self.value_proj = nn.Linear(low_rank_factor, hidden_size)
    
    def forward(self, hidden_states: torch.Tensor, prev_cache: torch.Tensor):
        key_states = self.key_proj(hidden_states)  # B x L x H
        key_states = nn.functional.normalize(key_states, p=2, dim=-1)
        value_states = self.value_proj(hidden_states) # B x L x H

        if prev_cache is not None:
            value_states = value_states - torch.einsum("b l h, b h d -> b l d", key_states, prev_cache)
            new_cache = prev_cache + torch.einsum("b l h, b l d -> b h d", key_states, value_states)
        else:
            new_cache = torch.einsum("b l h, b l d -> b h d", key_states, value_states)
        
        return new_cache