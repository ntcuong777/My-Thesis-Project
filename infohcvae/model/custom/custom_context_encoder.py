import torch
import torch.nn as nn
from infohcvae.model.custom.custom_lstm import CustomLSTM
from infohcvae.model.custom.luong_attention import LuongAttention
from infohcvae.model.custom.gated_self_attention import GatedAttention


class CustomContextEncoderForQG(nn.Module):
    def __init__(self, d_model, lstm_dec_nhidden, lstm_dec_nlayers, dropout=0.0):
        super(CustomContextEncoderForQG, self).__init__()

        self.context_lstm = CustomLSTM(input_size=d_model, hidden_size=lstm_dec_nhidden,
                                       num_layers=lstm_dec_nlayers, dropout=dropout,
                                       bidirectional=True)
        self.context_attention = GatedAttention(2 * lstm_dec_nhidden)

    def forward(self, c_a_embeds, c_mask, c_lengths):
        c_outputs, _ = self.context_lstm(c_a_embeds, c_lengths.to("cpu")) # c_outputs.size() = (N, len, hidden_size)

        # gated attention mechanism
        return self.context_attention(c_outputs, c_mask)
