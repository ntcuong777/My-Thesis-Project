import torch
import torch.nn as nn
from .luong_attention import LuongAttention
from .gated_self_attention import GatedAttention
from .custom_lstm import CustomLSTM


class BiLstmWithAttention(nn.Module):
    def __init__(self, d_model, lstm_hidden, lstm_layers, dropout=0.0, ):
        super().__init__()
        self.nhidden = lstm_hidden
        self.nlayers = lstm_layers
        self.bilstm = CustomLSTM(
            input_size=d_model, hidden_size=lstm_hidden, num_layers=lstm_layers,
            dropout=dropout, bidirectional=True)
        self.self_attention = GatedAttention(hidden_size=lstm_hidden * 2)
        self.final_state_attention = LuongAttention(lstm_hidden * 2, lstm_hidden * 2)

    def forward(self, input_embeds, input_lengths, attention_mask):
        input_hidden_states, input_states = self.bilstm(input_embeds, input_lengths.to("cpu"))
        input_hidden_states = self.self_attention(input_hidden_states, attention_mask=attention_mask)
        inp_h = input_states[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        inp_h = inp_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)
        # the final forward and reverse hidden states should attend to the whole sentence
        mask = attention_mask.unsqueeze(1)
        inp_h = self.final_state_attention(inp_h.unsqueeze(1), input_hidden_states, mask).squeeze(1)
        return input_hidden_states, inp_h
