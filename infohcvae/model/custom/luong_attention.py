import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttention(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear_proj = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights(self.linear_proj)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02) # N(0, 0.02)
            m.bias.data.fill_(0)

    def forward(self, query, memories, mask, return_attention_logits=None):
        # project query to the same hidden_size as memories
        query = self.linear_proj(query)

        # Luong attention computation
        mask = (1.0 - mask.float()) * (-3e4)
        # query (N, x, hidden_size), memories (N, len, hidden_size)
        attn_logits = torch.matmul(query, memories.transpose(-1, -2).contiguous()) # size = (N, x, len)
        attn_logits = attn_logits + mask
        attn_weights = F.softmax(attn_logits, dim=-1) # size = (N, x, len)
        attn_outputs = torch.matmul(attn_weights, memories) # size = (N, x, hidden_size)

        if return_attention_logits is not None and return_attention_logits:
            return attn_outputs, attn_logits
        return attn_outputs
