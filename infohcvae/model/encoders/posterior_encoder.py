import torch
import torch.nn as nn
from infohcvae.model.custom.luong_attention import LuongAttention
from infohcvae.model.custom.gated_self_attention import GatedAttention
from infohcvae.model.custom.custom_lstm import CustomLSTM
from infohcvae.model.model_utils import (
    gumbel_softmax, sample_gaussian,
    return_attention_mask, return_inputs_length
)


class PosteriorEncoder(nn.Module):
    def __init__(self, embedding, d_model, lstm_enc_nhidden, lstm_enc_nlayers,
                 nzqdim, nzadim, nza_values, dropout=0.0, pad_token_id=0):
        super(PosteriorEncoder, self).__init__()

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
        self.cq_self_attention = GatedAttention(2 * lstm_enc_nhidden)
        # self.cq_final_state_attention = LuongAttention(2 * lstm_enc_nhidden, 2 * lstm_enc_nhidden)

        self.context_answer_encoder = CustomLSTM(
            input_size=d_model, hidden_size=lstm_enc_nhidden, num_layers=lstm_enc_nlayers,
            dropout=dropout, bidirectional=True)
        self.ca_self_attention = GatedAttention(2 * lstm_enc_nhidden)
        # self.ca_final_state_attention = LuongAttention(2 * lstm_enc_nhidden, 2 * lstm_enc_nhidden)

        self.question_attention = LuongAttention(2 * lstm_enc_nhidden, 2 * lstm_enc_nhidden)
        self.context_attention = LuongAttention(2 * lstm_enc_nhidden, 2 * lstm_enc_nhidden)
        self.answer_zq_attention = LuongAttention(nzqdim, 2 * lstm_enc_nhidden)

        self.zq_mu_linear = nn.Linear(4 * 2 * lstm_enc_nhidden, nzqdim)
        self.zq_logvar_linear = nn.Linear(4 * 2 * lstm_enc_nhidden, nzqdim)
        self.za_linear = nn.Linear(nzqdim + 2 * 2 * lstm_enc_nhidden, nzadim * nza_values)

    def forward(self, c_ids, q_ids, a_mask):
        batch_size, _ = c_ids.size()
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_lengths = return_inputs_length(c_mask)

        q_mask = return_attention_mask(q_ids, self.pad_token_id)
        q_lengths = return_inputs_length(q_mask)

        # question enc
        q_embeds = self.embedding(q_ids)
        q_hidden_states, _ = self.context_question_encoder(q_embeds, q_lengths.to("cpu"))
        # q_h = q_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        # q_h = q_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)
        # question self attention
        q_hidden_states = self.cq_self_attention(q_hidden_states, q_mask)
        q_h_forward = q_hidden_states.view(batch_size, -1, 2, self.nhidden)[:, -1, 0, :]
        q_h_reverse = q_hidden_states.view(batch_size, -1, 2, self.nhidden)[:, 0, 1, :]
        q_h = torch.cat([q_h_forward, q_h_reverse], dim=1)
        # mask = q_mask.unsqueeze(1)
        # q_h = self.cq_final_state_attention(q_h.unsqueeze(1), q_hidden_states, mask).squeeze(1)

        # context enc
        c_embeds = self.embedding(c_ids)
        c_hidden_states, _ = self.context_question_encoder(c_embeds, c_lengths.to("cpu"))
        # c_h = c_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        # c_h = c_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)
        # context self attention
        c_hidden_states = self.cq_self_attention(c_hidden_states, c_mask)
        c_h_forward = c_hidden_states.view(batch_size, -1, 2, self.nhidden)[:, -1, 0, :]
        c_h_reverse = c_hidden_states.view(batch_size, -1, 2, self.nhidden)[:, 0, 1, :]
        c_h = torch.cat([c_h_forward, c_h_reverse], dim=1)
        # mask = c_mask.unsqueeze(1)
        # c_h = self.cq_final_state_attention(c_h.unsqueeze(1), c_hidden_states, mask).squeeze(1)

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

        # context and answer enc
        c_a_embeds = self.embedding(c_ids, a_mask, None)
        c_a_hidden_states, _ = self.context_answer_encoder(c_a_embeds, c_lengths.to("cpu"))
        # c_a_h = c_a_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        # c_a_h = c_a_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)
        # context-answer self-attention
        c_a_hidden_states = self.ca_self_attention(c_a_hidden_states, c_mask)
        c_a_h_forward = c_a_hidden_states.view(batch_size, -1, 2, self.nhidden)[:, -1, 0, :]
        c_a_h_reverse = c_a_hidden_states.view(batch_size, -1, 2, self.nhidden)[:, 0, 1, :]
        c_a_h = torch.cat([c_a_h_forward, c_a_h_reverse], dim=1)
        # mask = c_mask.unsqueeze(1)
        # c_a_h = self.ca_final_state_attention(c_a_h.unsqueeze(1), c_a_hidden_states, mask).squeeze(1)

        # attention zq, c_a
        mask = c_mask.unsqueeze(1)
        c_a_attned_by_zq = self.answer_zq_attention(zq.unsqueeze(1), c_a_hidden_states, mask).squeeze(1)

        h = torch.cat([zq, c_a_attned_by_zq, c_a_h], dim=-1)
        za_logits = self.za_linear(h).view(-1, self.nzadim, self.nza_values)
        # Sample `za`
        za = gumbel_softmax(za_logits, hard=True)

        return zq, zq_mu, zq_logvar, za, za_logits
