import torch
import torch.nn as nn
from infohcvae.model.custom.custom_lstm import CustomLSTM
from infohcvae.model.custom.luong_attention import LuongAttention
from infohcvae.model.model_utils import (
    return_inputs_length, return_attention_mask,
)


class CustomContextEncoderForQG(nn.Module):
    def __init__(self, embedding_model, d_model,
                 lstm_dec_nhidden, lstm_dec_nlayers, dropout=0.0):
        super(CustomContextEncoderForQG, self).__init__()

        self.embedding = embedding_model
        self.context_lstm = CustomLSTM(input_size=d_model, hidden_size=lstm_dec_nhidden,
                                       num_layers=lstm_dec_nlayers, dropout=dropout,
                                       bidirectional=True)
        self.context_attention = LuongAttention(2 * lstm_dec_nhidden, 2 * lstm_dec_nhidden)
        self.fusion = nn.Linear(4 * lstm_dec_nhidden, 2 * lstm_dec_nhidden, bias=False)
        self.gate = nn.Linear(4 * lstm_dec_nhidden, 2 * lstm_dec_nhidden, bias=False)

    def forward(self, c_ids, a_ids):
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_lengths = return_inputs_length(c_mask)

        c_embeddings = self.embedding(input_ids=c_ids, attention_mask=c_mask, token_type_ids=a_ids)
        c_outputs, _ = self.context_lstm(c_embeddings, c_lengths)

        # gated attention mechanism
        mask = torch.matmul(c_mask.unsqueeze(2), c_mask.unsqueeze(1))
        c_attned_by_c = self.context_attention(c_outputs, c_outputs, mask)
        c_concat = torch.cat([c_outputs, c_attned_by_c], dim=2)
        c_fused = self.fusion(c_concat).tanh()
        c_gate = self.gate(c_concat).sigmoid()
        c_outputs = c_gate * c_fused + (1 - c_gate) * c_outputs
        return c_outputs
