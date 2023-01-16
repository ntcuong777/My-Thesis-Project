import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.linear_proj = nn.Linear(in_features, out_features)

    def forward(self, query, memories, mask, return_attention_logits=None):
        # project query to the same hidden_size as memories
        query = self.linear_proj(query)

        # Luong attention computation
        mask = (1.0 - mask.float()) * -1000000.0
        # query (N, len, hidden_size), memories (N, len, hidden_size)
        attn_logits = torch.matmul(query, memories.transpose(-1, -2).contiguous())
        attn_logits = attn_logits + mask
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_outputs = torch.matmul(attn_weights, memories)

        if return_attention_logits is not None and return_attention_logits:
            return attn_outputs, attn_logits
        return attn_outputs
