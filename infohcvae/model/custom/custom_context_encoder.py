import torch
import torch.nn as nn
from infohcvae.model.custom.custom_lstm import CustomLSTM
from infohcvae.model.custom.gated_self_attention import GatedAttention
from infohcvae.model.model_utils import (
    return_attention_mask, return_inputs_length
)


class CustomContextEncoder(nn.Module):
    def __init__(self, embedding, d_model, lstm_dec_nhidden, lstm_dec_nlayers, dropout=0.0, pad_token_id=0):
        super(CustomContextEncoder, self).__init__()

        self.pad_token_id = pad_token_id

        self.embedding = embedding
        self.context_lstm = CustomLSTM(input_size=d_model, hidden_size=lstm_dec_nhidden,
                                       num_layers=lstm_dec_nlayers, dropout=dropout,
                                       bidirectional=True)
        self.context_attention = GatedAttention(2 * lstm_dec_nhidden)

    def forward(self, c_ids, c_a_mask):
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_lengths = return_inputs_length(c_mask)

        c_a_embeds = self.embedding(c_ids, c_mask, c_a_mask)
        c_outputs, _ = self.context_lstm(c_a_embeds, c_lengths.to("cpu")) # c_outputs.size() = (N, len, hidden_size)

        # gated attention mechanism
        return self.context_attention(c_outputs, c_mask)
