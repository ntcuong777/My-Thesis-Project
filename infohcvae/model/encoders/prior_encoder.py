import torch
import torch.nn as nn
from infohcvae.model.custom.luong_attention import LuongAttention
from infohcvae.model.custom.custom_lstm import CustomLSTM
from infohcvae.model.custom.bert_self_attention import BertSelfAttention
from infohcvae.model.model_utils import (
    gumbel_softmax, sample_gaussian,
)


class PriorEncoder(nn.Module):
    def __init__(self, d_model, lstm_enc_nhidden, lstm_enc_nlayers,
                 nzqdim, nzadim, nza_values, dropout=0, pad_token_id=0):
        super(PriorEncoder, self).__init__()

        self.pad_token_id = pad_token_id

        self.nhidden = lstm_enc_nhidden
        self.nlayers = lstm_enc_nlayers
        self.nzqdim = nzqdim
        self.nzadim = nzadim
        self.nza_values = nza_values

        self.context_encoder = CustomLSTM(input_size=d_model, hidden_size=lstm_enc_nhidden,
                                          num_layers=lstm_enc_nlayers, dropout=dropout,
                                          bidirectional=True)
        self.shared_self_attention = BertSelfAttention(hidden_size=lstm_enc_nhidden * 2, num_attention_heads=12)
        self.context_luong_attention = LuongAttention(2 * lstm_enc_nhidden, 2 * lstm_enc_nhidden)

        self.za_zq_attention = LuongAttention(nzqdim, 2 * lstm_enc_nhidden)

        self.zq_mu_linear = nn.Linear(2 * lstm_enc_nhidden, nzqdim)
        self.zq_logvar_linear = nn.Linear(2 * lstm_enc_nhidden, nzqdim)
        self.za_linear = nn.Linear(nzqdim + 2 * 2 * lstm_enc_nhidden, nzadim * nza_values)

    def forward(self, c_embeds, c_mask, c_lengths):
        c_hidden_states, c_state = self.context_encoder(c_embeds, c_lengths.to("cpu"))
        # skip connection
        c_hidden_states = c_hidden_states + self.shared_self_attention(c_hidden_states, attention_mask=c_mask)
        c_h = c_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_h = c_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)
        # the final forward and reverse hidden states should attend to the whole sentence
        mask = c_mask.unsqueeze(1)
        # skip connection
        c_h = c_h + self.context_luong_attention(c_h.unsqueeze(1), c_hidden_states, mask).squeeze(1)

        zq_mu = self.zq_mu_linear(c_h)
        zq_logvar = self.zq_logvar_linear(c_h)
        # Sample `zq`
        zq = sample_gaussian(zq_mu, zq_logvar)

        mask = c_mask.unsqueeze(1)
        c_attned_by_zq = self.za_zq_attention(zq.unsqueeze(1), c_hidden_states, mask).squeeze(1)
        c_attned_by_zq = c_attned_by_zq

        h = torch.cat([zq, c_attned_by_zq, c_h], dim=-1)
        za_logits = self.za_linear(h).view(-1, self.nzadim, self.nza_values)
        # sample `za`
        za = gumbel_softmax(za_logits, hard=True)

        return zq, zq_mu, zq_logvar, za, za_logits
