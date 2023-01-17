from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertAttention
from transformers import BertConfig


class CustomBertAttention(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, activation="gelu"):
        super().__init__()
        config = BertConfig.from_pretrained("bert-base-uncased") # Load arbitrary bert config
        config.hidden_size = hidden_size
        config.num_attention_heads = num_attention_heads
        config.hidden_act = activation
        self.attention = BertAttention(config)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
                use_cache: bool = None
    ):
        outputs = self.attention(hidden_states=hidden_states, attention_mask=attention_mask,
                                 past_key_value=past_key_value)
        if use_cache:
            return outputs # out = (hidden_states, past_key_values)

        return outputs[0] # hidden_states only