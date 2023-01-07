import copy

import torch
import torch.nn as nn
from infohcvae.model.model_utils import return_attention_mask, \
    cal_attn, gumbel_softmax, sample_gaussian
from transformers import T5EncoderModel, T5Config


class PosteriorEncoder(nn.Module):
    def __init__(self, pad_id, nzqdim, nzadim, nza_values, base_model="t5-base",
                 pooling_strategy="max", num_enc_finetune_layers=2):
        super(PosteriorEncoder, self).__init__()

        assert pooling_strategy in ["mean", "max"], \
            "The pooling strategy `%s` is not supported".format(pooling_strategy)

        config = T5Config.from_pretrained(base_model)

        self.t5_encoder = T5EncoderModel.from_pretrained(base_model)

        # Freeze all layers
        for param in self.t5_encoder.parameters():
            param.requires_grad = False
        # Only some top layers are for fine-tuning
        for idx in range(config.num_layers - num_enc_finetune_layers, config.num_layers):
            for param in self.t5_encoder.get_encoder().block[idx].parameters():
                param.requires_grad = True

        self.hidden_size = config.d_model
        self.nzqdim = nzqdim
        self.nzadim = nzadim
        self.nza_values = nza_values
        self.pad_token_id = pad_id
        self.pooling_strategy = pooling_strategy

        self.a_h_linear = nn.Linear(2*config.d_model, config.d_model)
        self.zq_attention = nn.Linear(nzqdim, config.d_model)

        self.zq_mu_linear = nn.Linear(4*config.d_model, nzqdim, bias=False)
        self.zq_logvar_linear = nn.Linear(4*config.d_model, nzqdim, bias=False)
        self.za_linear = nn.Linear(2*config.d_model + nzqdim, nzadim * nza_values, bias=False)

    def pool(self, x):
        # Shape of x - (layer_count, batch_size, seq_length, hidden_size)
        x = torch.stack(x[1:])
        x = x.transpose(0, 1)
        if self.pooling_strategy == "mean":
            return x[:, -1, :, :].mean(dim=1)
        elif self.pooling_strategy == "max":
            return torch.max(x[:, -1, :, :], dim=1)[0]  # Pool from last layer.
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

    def forward(self, c_ids, q_ids, a_mask):
        N, _ = c_ids.size()
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        q_mask = return_attention_mask(q_ids, self.pad_token_id)

        # question enc
        q_hidden_states = self.t5_encoder(input_ids=q_ids, attention_mask=q_mask)[0]

        # context enc
        c_hidden_states = self.t5_encoder(input_ids=c_ids, attention_mask=c_mask)[0]

        # context and answer enc
        a_ids = c_ids * a_mask
        a_hidden_states = self.t5_encoder(input_ids=a_ids, attention_mask=a_mask)[0]
        a_attned_by_c, _ = cal_attn(a_hidden_states, c_hidden_states, c_mask.unsqueeze(1)) # output shape = (N, seq_len, d_model)
        c_attned_by_a, _ = cal_attn(c_hidden_states, a_hidden_states, a_mask.unsqueeze(1)) # output shape = (N, seq_len, d_model)
        c_a_hidden_states = self.a_h_linear(torch.cat([a_attned_by_c, c_attned_by_a], dim=-1))

        # attetion q, c
        mask = c_mask.unsqueeze(1)
        c_attned_by_q, _ = cal_attn(q_hidden_states, c_hidden_states, mask) # output shape = (N, seq_len, d_model)

        # attetion c, q
        mask = q_mask.unsqueeze(1)
        q_attned_by_c, _ = cal_attn(c_hidden_states, q_hidden_states, mask) # output shape = (N, seq_len, d_model)

        # `h`'s shape = (N, seq_len, 4*d_model)
        h = torch.cat([q_hidden_states, q_attned_by_c, c_hidden_states, c_attned_by_q], dim=-1)
        pooled = self.pool(h)
        zq, zq_mu, zq_logvar = self.calculate_zq_latent(pooled)

        # attention zq, c_a
        mask = c_mask.unsqueeze(1)
        c_a_attned_by_zq, _ = cal_attn(self.zq_attention(zq).unsqueeze(1),
                                       c_a_hidden_states, mask) # output shape = (N, 1, d_model)
        c_a_attned_by_zq = c_a_attned_by_zq.squeeze(1)

        h = torch.cat([zq, c_a_attned_by_zq, self.pool(c_a_hidden_states)], dim=-1)
        za, za_logits = self.calculate_za_latent(h, hard=True)

        if self.training:
            return zq_mu, zq_logvar, zq, za_logits, za
        else:
            return zq, za
