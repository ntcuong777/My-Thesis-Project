import torch
import torch.nn as nn
from infohcvae.model.custom.luong_attention import LuongAttention
from infohcvae.model.custom.custom_lstm import CustomLSTM
from infohcvae.model.model_utils import (
    gumbel_softmax, return_inputs_length,
    return_attention_mask, sample_gaussian,
)


class PosteriorEncoder(nn.Module):
    def __init__(self, embedding_model, d_model,
                 lstm_enc_nhidden, lstm_enc_nlayers,
                 nzqdim, nzadim, nza_values, dropout=0.0, pad_token_id=0):
        super(PosteriorEncoder, self).__init__()

        self.pad_token_id = pad_token_id
        self.embedding = embedding_model
        self.nhidden = lstm_enc_nhidden
        self.nlayers = lstm_enc_nlayers
        self.nzqdim = nzqdim
        self.nzadim = nzadim
        self.nza_values = nza_values

        self.answer_encoder = CustomLSTM(input_size=d_model, hidden_size=lstm_enc_nhidden,
                                         num_layers=lstm_enc_nlayers, dropout=dropout,
                                         bidirectional=True)
        self.question_encoder = CustomLSTM(input_size=d_model, hidden_size=lstm_enc_nhidden,
                                           num_layers=lstm_enc_nlayers, dropout=dropout,
                                           bidirectional=True)
        self.context_encoder = CustomLSTM(input_size=d_model, hidden_size=lstm_enc_nhidden,
                                          num_layers=lstm_enc_nlayers, dropout=dropout,
                                          bidirectional=True)

        self.question_attention = LuongAttention(2 * lstm_enc_nhidden, 2 * lstm_enc_nhidden)
        self.context_attention = LuongAttention(2 * lstm_enc_nhidden, 2 * lstm_enc_nhidden)
        self.answer_zq_attention = LuongAttention(nzqdim, 2 * lstm_enc_nhidden)

        self.zq_mu_linear = nn.Linear(4 * 2 * lstm_enc_nhidden, nzqdim)
        self.zq_logvar_linear = nn.Linear(4 * 2 * lstm_enc_nhidden, nzqdim)
        self.za_linear = nn.Linear(nzqdim + 2 * 2 * lstm_enc_nhidden, nza_values * nza_values)

    def forward(self, c_ids, q_ids, c_a_mask):
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_lengths = return_inputs_length(c_mask)
        q_mask = return_attention_mask(q_ids, self.pad_token_id)
        q_lengths = return_inputs_length(q_ids)

        # question enc
        q_embeddings = self.embedding(input_ids=q_ids, attention_mask=q_mask)[0]
        q_hidden_states, q_state = self.question_encoder(q_embeddings, q_lengths)
        q_h = q_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        q_h = q_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        # context enc
        c_embeddings = self.embedding(input_ids=c_ids, attention_mask=c_mask)[0]
        c_hidden_states, c_state = self.context_encoder(c_embeddings, c_lengths)
        c_h = c_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_h = c_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        # context and answer enc
        c_a_embeddings = self.embedding(input_ids=c_ids, token_type_ids=c_a_mask, attention_mask=c_mask)[0]
        c_a_hidden_states, c_a_state = self.answer_encoder(c_a_embeddings, c_lengths)
        c_a_h = c_a_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_a_h = c_a_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        # attetion q, c
        mask = c_mask.unsqueeze(1)
        c_attned_by_q = self.question_attention(q_h.unsqueeze(1), c_hidden_states, mask).squeeze(1)

        # attetion c, q
        mask = q_mask.unsqueeze(1)
        q_attned_by_c = self.context_attention(c_h.unsqueeze(1), q_hidden_states, mask).squeeze(1)

        h = torch.cat([q_h, q_attned_by_c, c_h, c_attned_by_q], dim=-1)
        zq_mu = self.zq_mu_linear(h)
        zq_logvar = self.zq_logvar_linear(h)
        # Sample `zq`
        zq = sample_gaussian(zq_mu, zq_logvar)

        # attention zq, c_a
        mask = c_mask.unsqueeze(1)
        c_a_attned_by_zq = self.answer_zq_attention(zq.unsqueeze(1), c_a_hidden_states, mask).squeeze(1)

        h = torch.cat([zq, c_a_attned_by_zq, c_a_h], dim=-1)
        za_logits = self.za_linear(h).view(-1, self.nzadim, self.nza_values)
        # Sample `za`
        za = gumbel_softmax(za_logits, hard=True)

        return zq, zq_mu, zq_logvar, za, za_logits
