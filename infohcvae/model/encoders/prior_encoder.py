import torch
import torch.nn as nn
from infohcvae.model.custom.luong_attention import LuongAttention
from infohcvae.model.custom.bilstm_with_attention import BiLstmWithAttention
from infohcvae.model.model_utils import (
    gumbel_softmax, sample_gaussian,
    return_attention_mask, return_inputs_length
)


class PriorEncoder(nn.Module):
    def __init__(self, embedding, d_model, lstm_enc_nhidden, lstm_enc_nlayers,
                 nzqdim, nzadim, nza_values, dropout=0, pad_token_id=0):
        super(PriorEncoder, self).__init__()

        self.embedding = embedding

        self.pad_token_id = pad_token_id

        self.nhidden = lstm_enc_nhidden
        self.nlayers = lstm_enc_nlayers
        self.nzqdim = nzqdim
        self.nzadim = nzadim
        self.nza_values = nza_values

        self.context_question_encoder = BiLstmWithAttention(
            d_model, lstm_enc_nhidden, lstm_enc_nlayers, dropout=dropout)
        self.context_answer_encoder = BiLstmWithAttention(
            d_model, lstm_enc_nhidden, lstm_enc_nlayers, dropout=dropout)

        self.answer_zq_attention = LuongAttention(nzqdim, 2 * lstm_enc_nhidden)

        self.zq_mu_linear = nn.Linear(2 * lstm_enc_nhidden, nzqdim)
        self.zq_logvar_linear = nn.Linear(2 * lstm_enc_nhidden, nzqdim)
        self.za_linear = nn.Linear(nzqdim + 2 * 2 * lstm_enc_nhidden, nzadim * nza_values)

    def forward(self, c_ids):
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_lengths = return_inputs_length(c_mask)

        c_embeds = self.embedding(c_ids)
        cq_hidden_states, cq_h = self.context_question_encoder(c_embeds, c_lengths, c_mask)
        ca_hidden_states, ca_h = self.context_answer_encoder(c_embeds, c_lengths, c_mask)

        zq_mu = self.zq_mu_linear(cq_h)
        zq_logvar = self.zq_logvar_linear(cq_h)
        # Sample `zq`
        zq = sample_gaussian(zq_mu, zq_logvar)

        mask = c_mask.unsqueeze(1)
        c_attned_by_zq = self.answer_zq_attention(zq.unsqueeze(1), ca_hidden_states, mask).squeeze(1)

        h = torch.cat([zq, c_attned_by_zq, ca_h], dim=-1)
        za_logits = self.za_linear(h).view(-1, self.nzadim, self.nza_values)
        # sample `za`
        za = gumbel_softmax(za_logits, hard=True)

        return zq, zq_mu, zq_logvar, za, za_logits
