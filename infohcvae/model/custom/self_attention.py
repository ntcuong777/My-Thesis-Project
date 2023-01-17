from typing import Optional, Tuple

import torch
import torch.nn as nn
from .multihead_attention import AddNormWithMultiHeadAttention

class SelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12):
        super().__init__()
        self.attention = AddNormWithMultiHeadAttention(
            query_in_features=hidden_size, key_in_features=hidden_size, value_in_features=hidden_size,
            out_features=hidden_size, num_heads=num_attention_heads)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None,):
        return self.attention(hidden_states, hidden_states, hidden_states, hidden_states, mask=attention_mask)
