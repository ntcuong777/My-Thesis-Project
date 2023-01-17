import torch
import torch.nn as nn
from infohcvae.model.custom.custom_lstm import CustomLSTM
from infohcvae.model.custom.bert_self_attention import BertSelfAttention


class CustomContextEncoderForQG(nn.Module):
    def __init__(self, d_model, lstm_dec_nhidden, lstm_dec_nlayers, dropout=0.0):
        super(CustomContextEncoderForQG, self).__init__()

        self.context_lstm = CustomLSTM(input_size=d_model, hidden_size=lstm_dec_nhidden,
                                       num_layers=lstm_dec_nlayers, dropout=dropout,
                                       bidirectional=True)
        self.context_attention = BertSelfAttention(2 * lstm_dec_nhidden, num_attention_heads=10)
        # self.fusion = nn.Linear(4 * lstm_dec_nhidden, 2 * lstm_dec_nhidden, bias=False)
        # self.gate = nn.Linear(4 * lstm_dec_nhidden, 2 * lstm_dec_nhidden, bias=False)

    def forward(self, c_a_embeds, c_mask, c_lengths):
        c_outputs, _ = self.context_lstm(c_a_embeds, c_lengths)

        # skip connection with self attention
        c_outputs = c_outputs + self.context_attention(c_outputs, attention_mask=c_mask)

        # gated attention mechanism
        # mask = torch.matmul(c_mask.unsqueeze(2), c_mask.unsqueeze(1))
        # c_attned_by_c = c_outputs + self.context_attention(c_outputs, attention_mask=c_mask)
        # c_concat = torch.cat([c_outputs, c_attned_by_c], dim=2)
        # c_fused = self.fusion(c_concat).tanh()
        # c_gate = self.gate(c_concat).sigmoid()
        # c_outputs = c_gate * c_fused + (1 - c_gate) * c_outputs
        return c_outputs
