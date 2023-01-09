import copy

import torch
import torch.nn as nn
from infohcvae.model.model_utils import return_attention_mask, \
    cal_attn, gumbel_softmax, sample_gaussian
from transformers import T5Config
from t5_context_answer_encoder import T5ContextAnswerEncoder


class PosteriorEncoder(nn.Module):
    def __init__(self, pad_id, nzqdim, nzadim, nza_values, base_model="t5-base",
                 pooling_strategy="max", num_enc_finetune_layers=2):
        super(PosteriorEncoder, self).__init__()

        assert pooling_strategy in ["mean", "max"], \
            "The pooling strategy `%s` is not supported".format(pooling_strategy)

        config = T5Config.from_pretrained(base_model)

        self.encoder = T5ContextAnswerEncoder(base_model=base_model, num_enc_finetune_layers=num_enc_finetune_layers)

        self.hidden_size = config.d_model
        self.nzqdim = nzqdim
        self.nzadim = nzadim
        self.nza_values = nza_values
        self.pad_token_id = pad_id
        self.pooling_strategy = pooling_strategy

        self.za_attention = nn.Linear(nzqdim, config.d_model, bias=False)

        self.zq_mu_linear = nn.Linear(6*config.d_model + nzadim * nza_values, nzqdim, bias=False)
        self.zq_logvar_linear = nn.Linear(6*config.d_model + nzadim * nza_values, nzqdim, bias=False)
        self.za_linear = nn.Linear(config.d_model, nzadim * nza_values, bias=False)

    def get_context_answer_t5_encoder(self):
        return self.encoder

    def pool(self, last_hidden_states):
        # last_hidden_states shape = (batch_size, seq_length, hidden_size)
        # pooling over `seq_length` dim
        if self.pooling_strategy == "mean":
            return last_hidden_states.mean(dim=1)
        elif self.pooling_strategy == "max":
            return torch.max(last_hidden_states, dim=1)[0]  # Pool from last layer
        else:
            raise Exception("Wrong pooling strategy!")

    def calculate_zq_latent(self, pooled):
        zq_mu, zq_logvar = self.zq_mu_linear(pooled), self.zq_logvar_linear(pooled)
        zq = sample_gaussian(zq_mu, zq_logvar)
        return zq, zq_mu, zq_logvar

    def calculate_za_latent(self, pooled, hard=True):
        za_logits = self.za_linear(pooled).view(-1, self.nzadim, self.nza_values)
        za = gumbel_softmax(za_logits, hard=hard)
        return za, za_logits

    def forward(self, c_ids, q_ids, a_mask, return_distribution_parameters=None):
        N, _ = c_ids.size()
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        q_mask = return_attention_mask(q_ids, self.pad_token_id)

        # context enc
        c_hidden_states, c_a_hidden_states = self.encoder(context_ids=c_ids, context_mask=c_mask, answer_mask=a_mask)

        # sample `za`
        h = self.pool(c_a_hidden_states)
        za, za_logits = self.calculate_za_latent(h, hard=True)

        # question enc
        q_hidden_states = self.t5_encoder(input_ids=q_ids, attention_mask=q_mask)[0]

        # attetion q, c
        mask = torch.matmul(q_mask.unsqueeze(2), c_mask.unsqueeze(1))
        q_attned_to_c, _ = cal_attn(q_hidden_states, c_hidden_states, mask)  # output shape = (N, seq_len, d_model)

        # attetion c, q
        mask = torch.matmul(c_mask.unsqueeze(2), q_mask.unsqueeze(1))
        c_attned_to_q, _ = cal_attn(c_hidden_states, q_hidden_states, mask)  # output shape = (N, seq_len, d_model)

        # attention za, q_attned_to_c
        mask = q_mask.unsqueeze(1)
        za_attned_to_q, _ = cal_attn(self.za_attention(za.view(N, -1)).unsqueeze(1),
                                     q_attned_to_c, mask)  # output shape = (N, 1, d_model)
        za_attned_to_q = za_attned_to_q.squeeze(1)

        # attention za, c_attned_to_q
        mask = c_mask.unsqueeze(1)
        za_attned_to_c, _ = cal_attn(self.za_attention(za.view(N, -1)).unsqueeze(1),
                                     c_attned_to_q, mask)  # output shape = (N, 1, d_model)
        za_attned_to_c = za_attned_to_c.squeeze(1)

        # sample `zq`
        # `h`'s shape = (N, seq_len, 3*d_model)
        hq = torch.cat([q_hidden_states, za_attned_to_q, q_attned_to_c], dim=-1)
        hc = torch.cat([c_hidden_states, za_attned_to_c, c_attned_to_q], dim=-1)
        # pooled outputs has shape = (N, 3*d_model)
        pooled_q, pooled_c = self.pool(hq), self.pool(hc)
        zq, zq_mu, zq_logvar = self.calculate_zq_latent(torch.cat([za.view(N, -1), pooled_q, pooled_c], dim=-1))

        if return_distribution_parameters is not None and return_distribution_parameters:
            return zq_mu, zq_logvar, zq, za_logits, za
        else:
            return zq, za
