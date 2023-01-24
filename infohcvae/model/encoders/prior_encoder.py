import torch
import torch.nn as nn
from infohcvae.model.custom.luong_attention import LuongAttention
from infohcvae.model.custom.gated_self_attention import GatedAttention
from infohcvae.model.custom.custom_lstm import CustomLSTM
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

        self.context_question_encoder = CustomLSTM(
            input_size=d_model, hidden_size=lstm_enc_nhidden, num_layers=lstm_enc_nlayers,
            dropout=dropout, bidirectional=True)
        self.cq_self_attention = GatedAttention(lstm_enc_nhidden)
        # self.cq_final_state_attention = LuongAttention(2 * lstm_enc_nhidden, 2 * lstm_enc_nhidden)

        self.context_answer_encoder = CustomLSTM(
            input_size=d_model, hidden_size=lstm_enc_nhidden, num_layers=lstm_enc_nlayers,
            dropout=dropout, bidirectional=True)
        self.ca_self_attention = GatedAttention(lstm_enc_nhidden)
        # self.ca_final_state_attention = LuongAttention(2 * lstm_enc_nhidden, 2 * lstm_enc_nhidden)

        self.answer_zq_attention = LuongAttention(nzqdim, 2 * lstm_enc_nhidden)

        self.zq_mu_linear = nn.Linear(2 * lstm_enc_nhidden, nzqdim)
        self.zq_logvar_linear = nn.Linear(2 * lstm_enc_nhidden, nzqdim)
        self.za_linear = nn.Linear(nzqdim + 2 * 2 * lstm_enc_nhidden, nzadim * nza_values)

    def forward(self, c_ids):
        batch_size, _ = c_ids.size()
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_lengths = return_inputs_length(c_mask)

        c_embeds = self.embedding(c_ids)
        cq_hs, _ = self.context_question_encoder(c_embeds, c_lengths.to("cpu"))
        cq_hs = cq_hs.view(batch_size, -1, 2, self.nhidden)
        cq_fwd_hs = cq_hs[:, :, 0, :]
        cq_rev_hs = cq_hs[:, :, 1, :]
        # question encoder self attention
        cq_fwd_hs = self.cq_self_attention(cq_fwd_hs, c_mask)
        cq_rev_hs = self.cq_self_attention(cq_rev_hs, c_mask)
        cq_h = torch.cat([cq_fwd_hs[:, -1, :], cq_rev_hs[:, 0, :]], dim=1)

        ca_hs, _ = self.context_answer_encoder(c_embeds, c_lengths.to("cpu"))
        ca_hs = ca_hs.view(batch_size, -1, 2, self.nhidden)
        ca_fwd_hs = ca_hs[:, :, 0, :]
        ca_rev_hs = ca_hs[:, :, 1, :]
        # context-answer self-attention
        ca_fwd_hs = self.ca_self_attention(ca_fwd_hs, c_mask)
        ca_rev_hs = self.ca_self_attention(ca_rev_hs, c_mask)
        # re-concat
        ca_hs = torch.cat(
            [ca_fwd_hs.unsqueeze(2), ca_rev_hs.unsqueeze(2)], dim=2).view(batch_size, -1, 2 * self.nhidden)
        ca_h = torch.cat([ca_fwd_hs[:, -1, :], ca_rev_hs[:, 0, :]], dim=1)

        zq_mu = self.zq_mu_linear(cq_h)
        zq_logvar = self.zq_logvar_linear(cq_h)
        # Sample `zq`
        zq = sample_gaussian(zq_mu, zq_logvar)

        mask = c_mask.unsqueeze(1)
        c_attned_by_zq = self.answer_zq_attention(zq.unsqueeze(1), ca_hs, mask).squeeze(1)

        h = torch.cat([zq, c_attned_by_zq, ca_h], dim=-1)
        za_logits = self.za_linear(h).view(-1, self.nzadim, self.nza_values)
        # sample `za`
        za = gumbel_softmax(za_logits, hard=True)

        return zq, zq_mu, zq_logvar, za, za_logits
