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

        self.encoder = CustomLSTM(
            input_size=d_model, hidden_size=lstm_enc_nhidden, num_layers=lstm_enc_nlayers,
            dropout=dropout, bidirectional=True)
        # self.self_attention = GatedAttention(2 * lstm_enc_nhidden)
        # self.final_state_attention = LuongAttention(2 * lstm_enc_nhidden, 2 * lstm_enc_nhidden)

        self.answer_zq_attention = LuongAttention(nzqdim, 2 * lstm_enc_nhidden)

        self.zq_mu_linear = nn.Sequential(
            nn.Linear(2 * lstm_enc_nhidden, nzqdim),
            nn.BatchNorm1d(nzqdim, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(nzqdim, nzqdim),
            nn.BatchNorm1d(nzqdim, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(nzqdim, nzqdim)
        )
        self.zq_logvar_linear = nn.Sequential(
            nn.Linear(2 * lstm_enc_nhidden, nzqdim),
            nn.BatchNorm1d(nzqdim, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(nzqdim, nzqdim),
            nn.BatchNorm1d(nzqdim, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(nzqdim, nzqdim)
        )
        self.zq_generator = nn.Sequential(
            nn.Linear(nzqdim, nzqdim),
            nn.BatchNorm1d(nzqdim, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(nzqdim, nzqdim),
            nn.BatchNorm1d(nzqdim, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(nzqdim, nzqdim)
        )
        self.zq_mu_linear.apply(self.init_weights)
        self.zq_logvar_linear.apply(self.init_weights)
        self.zq_generator.apply(self.init_weights)

        self.za_linear = nn.Sequential(
            nn.Linear(nzqdim + 2 * 2 * lstm_enc_nhidden, nzadim * nza_values),
            nn.BatchNorm1d(nzadim * nza_values, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(nzadim * nza_values, nzadim * nza_values),
            nn.BatchNorm1d(nzadim * nza_values, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(nzadim * nza_values, nzadim * nza_values)
        )
        self.za_generator = nn.Sequential(
            nn.Linear(nzadim * nza_values, nzadim * nza_values),
            nn.BatchNorm1d(nzadim * nza_values, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(nzadim * nza_values, nzadim * nza_values),
            nn.BatchNorm1d(nzadim * nza_values, eps=1e-05, momentum=0.1),
            nn.ReLU(),
            nn.Linear(nzadim * nza_values, nzadim * nza_values)
        )
        self.za_linear.apply(self.init_weights)
        self.za_generator.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02) # N(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, c_ids):
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_lengths = return_inputs_length(c_mask)

        c_embeds = self.embedding(c_ids)
        c_hs, c_states = self.encoder(c_embeds, c_lengths.to("cpu"))
        c_h = c_states[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_h = c_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        zq_mu = self.zq_mu_linear(c_h)
        zq_logvar = self.zq_logvar_linear(c_h)
        # Sample `zq`
        zq = self.zq_generator(sample_gaussian(zq_mu, zq_logvar))

        mask = c_mask.unsqueeze(1)
        c_attned_by_zq = self.answer_zq_attention(zq.unsqueeze(1), c_hs, mask).squeeze(1)

        h = torch.cat([zq, c_attned_by_zq, c_h], dim=-1)
        za_logits = self.za_linear(h).view(-1, self.nzadim, self.nza_values)
        # sample `za`
        za = gumbel_softmax(za_logits, hard=True)
        za = self.za_generator(za.view(-1, self.nzadim * self.nza_values)).view(-1, self.nzadim, self.nza_values)

        return zq, zq_mu, zq_logvar, za, za_logits
