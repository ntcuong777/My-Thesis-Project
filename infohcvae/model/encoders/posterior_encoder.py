import torch
import torch.nn as nn
from infohcvae.model.model_utils import return_attention_mask, \
    gumbel_softmax, sample_gaussian
from transformers import T5Config


class PosteriorQAEncoder(nn.Module):
    def __init__(self, pad_id, encoder_net, nzqdim, nzadim, nza_values, base_model="t5-base",
                 pooling_strategy="max", num_enc_finetune_layers=3):
        super(PosteriorQAEncoder, self).__init__()

        assert pooling_strategy in ["mean", "max"], \
            "The pooling strategy `%s` is not supported".format(pooling_strategy)

        config = T5Config.from_pretrained(base_model)

        # Common encoder for both question and answer encoder
        self.encoder = encoder_net

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

        """ START: Answer encoding to get latent `za` """
        # context enc
        c_a_hidden_states = self.encoder(context_ids=c_ids, context_mask=c_mask, answer_mask=a_mask)

        # sample `za`
        za, za_logits = self.calculate_za_latent(self.pool(c_a_hidden_states), hard=True)
        """ END: Answer encoding to get latent `za` """

        """ START: Question encoding to get latent `zq` """
        # The context has its own [CLS] token at index 0, we need to leave that out
        qc_ids = torch.cat([q_ids, c_ids[:, 1:]], dim=-1)
        qc_mask = torch.cat([q_mask, c_mask[:, 1:]], dim=-1)
        qc_a_mask = torch.cat([q_mask, a_mask[:, 1:]], dim=-1)
        # question enc
        qc_a_hidden_states = self.encoder(input_ids=qc_ids, attention_mask=qc_mask, answer_mask=qc_a_mask)

        # sample `zq`
        # pooled outputs has shape = (N, d_model)
        zq, zq_mu, zq_logvar = self.calculate_zq_latent(self.pool(qc_a_hidden_states))
        """ END: Question encoding to get latent `zq` """

        if return_distribution_parameters is not None and return_distribution_parameters:
            return za_logits, za, zq_mu, zq_logvar, zq
        else:
            return za, zq
