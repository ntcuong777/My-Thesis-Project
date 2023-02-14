import torch
import torch.nn as nn
from .luong_attention import LuongAttention
from .gated_self_attention import GatedAttention
from .custom_lstm import CustomLSTM


class AnswerDecoderBiLstmWithAttention(nn.Module):
    def __init__(self, d_model, lstm_hidden, lstm_layers, dropout=0.0):
        super().__init__()
        self.nhidden = lstm_hidden
        self.nlayers = lstm_layers
        self.bilstm = CustomLSTM(
            input_size=d_model, hidden_size=lstm_hidden, num_layers=lstm_layers,
            dropout=dropout, bidirectional=True)
        self.self_attention = GatedAttention(hidden_size=lstm_hidden * 2)

    def forward(self, input_embeds, input_lengths, attention_mask, q_init_state=None):
        input_hs, _ = self.bilstm(input_embeds, input_lengths.to("cpu"), state=q_init_state)
        input_hs = self.self_attention(input_hs, attention_mask=attention_mask)
        return input_hs
