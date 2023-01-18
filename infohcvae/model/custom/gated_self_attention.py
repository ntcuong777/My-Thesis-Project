import torch
import torch.nn as nn
from infohcvae.model.custom.luong_attention import LuongAttention


class GatedAttention(nn.Module):
    def __init__(self, hidden_size=768):
        super(GatedAttention, self).__init__()
        self.context_attention = LuongAttention(hidden_size, hidden_size)
        self.fusion = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.gate = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask):
        mask = torch.matmul(attention_mask.unsqueeze(2), attention_mask.unsqueeze(1))

        # gated attention mechanism
        attned_states = self.context_attention(hidden_states, hidden_states, mask)
        hs_concat = torch.cat([hidden_states, attned_states], dim=2)
        hs_fused = self.fusion(hs_concat).tanh()
        hs_gate = self.gate(hs_concat).sigmoid()
        hs_outputs = hs_gate * hs_fused + (1 - hs_gate) * hidden_states
        return hs_outputs
