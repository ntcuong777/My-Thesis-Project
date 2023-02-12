import torch
import torch.nn as nn
from infohcvae.model.custom.luong_attention import LuongAttention
from infohcvae.model.custom.custom_lstm import CustomLSTM
from infohcvae.model.model_utils import (
    gumbel_softmax, sample_gaussian,
    return_attention_mask, return_inputs_length
)


class PosteriorEncoder(nn.Module):
    def __init__(self, embedding, d_model, lstm_enc_nhidden, lstm_enc_nlayers,
                 nzqdim, nzadim, dropout=0.0, pad_token_id=0):
        super(PosteriorEncoder, self).__init__()

        self.embedding = embedding

        self.pad_token_id = pad_token_id
        self.nhidden = lstm_enc_nhidden
        self.nlayers = lstm_enc_nlayers
        self.nzqdim = nzqdim
        self.nzadim = nzadim
        # self.nza_values = nza_values

        self.encoder = CustomLSTM(
            input_size=d_model, hidden_size=lstm_enc_nhidden, num_layers=lstm_enc_nlayers,
            dropout=dropout, bidirectional=True)

        self.question_attention = LuongAttention(2 * lstm_enc_nhidden, 2 * lstm_enc_nhidden)
        self.context_attention = LuongAttention(2 * lstm_enc_nhidden, 2 * lstm_enc_nhidden)
        self.answer_zq_attention = LuongAttention(nzqdim, 2 * lstm_enc_nhidden)

        self.zq_linear = nn.Sequential(
            nn.Linear(4 * 2 * lstm_enc_nhidden, 2 * nzqdim),
            nn.BatchNorm1d(2 * nzqdim, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(2 * nzqdim, 2 * nzqdim),
            nn.BatchNorm1d(2 * nzqdim, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(2 * nzqdim, 2 * nzqdim)
        )
        self.zq_linear.apply(self.init_weights)

        self.za_linear = nn.Sequential(
            nn.Linear(nzqdim + 2 * 2 * lstm_enc_nhidden, 2 * nzadim),
            nn.BatchNorm1d(2 * nzadim, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(2 * nzadim, 2 * nzadim),
            nn.BatchNorm1d(2 * nzadim, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(2 * nzadim, 2 * nzadim)
        )
        self.za_linear.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02) # N(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, c_ids, q_ids, a_mask):
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_lengths = return_inputs_length(c_mask)

        q_mask = return_attention_mask(q_ids, self.pad_token_id)
        q_lengths = return_inputs_length(q_mask)

        # question enc
        q_embeds = self.embedding(q_ids)
        q_hidden_states, q_state = self.encoder(q_embeds, q_lengths.to("cpu"))
        q_h = q_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        q_h = q_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        # context enc
        c_embeds = self.embedding(c_ids)
        c_hidden_states, c_state = self.encoder(c_embeds, c_lengths.to("cpu"))
        c_h = c_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_h = c_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        # attetion q, c
        mask = c_mask.unsqueeze(1)
        c_attned_by_q = self.question_attention(q_h.unsqueeze(1), c_hidden_states, mask).squeeze(1)

        # attetion c, q
        mask = q_mask.unsqueeze(1)
        q_attned_by_c = self.context_attention(c_h.unsqueeze(1), q_hidden_states, mask).squeeze(1)

        h = torch.cat([q_h, q_attned_by_c, c_h, c_attned_by_q], dim=-1)
        zq_params = self.zq_linear(h)
        zq_mu, zq_logvar = torch.split(zq_params, self.nzqdim, dim=-1)
        # Sample `zq`
        zq = sample_gaussian(zq_mu, zq_logvar)

        # context and answer enc
        c_a_embeds = self.embedding(c_ids, a_mask, None)
        c_a_hidden_states, c_a_state = self.encoder(c_a_embeds, c_lengths.to("cpu"))
        c_a_h = c_a_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_a_h = c_a_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        # attention zq, c_a
        mask = c_mask.unsqueeze(1)
        c_a_attned_by_zq = self.answer_zq_attention(zq.unsqueeze(1), c_a_hidden_states, mask).squeeze(1)

        h = torch.cat([zq, c_a_attned_by_zq, c_a_h], dim=-1)
        za_params = self.za_linear(h)
        za_mu, za_logvar = torch.split(za_params, self.nzadim, dim=-1)
        # Sample `za`
        za = sample_gaussian(za_mu, za_logvar)

        return zq, zq_mu, zq_logvar, za, za_mu, za_logvar
