import torch
import torch.nn as nn
from infohcvae.model.custom.luong_attention import LuongAttention


class GatedAttention(nn.Module):
    def __init__(self, hidden_size=768):
        super(GatedAttention, self).__init__()
        self.context_attention = LuongAttention(hidden_size, hidden_size)
        self.fusion = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.gate = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        self.init_weights(self.fusion)
        self.init_weights(self.gate)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02) # N(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, hidden_states, attention_mask):
        extended_attention_mask = attention_mask
        if len(attention_mask.size()) == 2: # mask shape = (N, seq_len) => expand to attention matrix mask
            extended_attention_mask = torch.matmul(attention_mask.unsqueeze(2), attention_mask.unsqueeze(1))
        else:
            assert len(attention_mask.size()) == 3, \
                "Wrong number of dimension of `attention_mask`, " + \
                "the shape should either be (N, seq_len) or (N, seq_len, seq_len) where N is the batch size"

        # gated attention mechanism
        attned_states = self.context_attention(hidden_states, hidden_states, extended_attention_mask)
        hs_concat = torch.cat([hidden_states, attned_states], dim=2)
        hs_fused = self.fusion(hs_concat).tanh()
        hs_gate = self.gate(hs_concat).sigmoid()
        hs_outputs = hs_gate * hs_fused + (1 - hs_gate) * hidden_states
        return hs_outputs
