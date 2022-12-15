import torch
import torch.nn as nn
from infohcvae.model.model_utils import return_attention_mask, \
    cal_attn, gumbel_softmax, sample_gaussian, softargmax


class PosteriorEncoder(nn.Module):
    def __init__(self, pad_id, context_enc, hidden_size,
                 nzqdim, nza, nzadim, n_enc_layers, dropout=0.0):
        super(PosteriorEncoder, self).__init__()

        self.context_encoder = context_enc
        self.hidden_size = hidden_size
        self.nzqdim = nzqdim
        self.nza = nza
        self.nzadim = nzadim
        self.pad_token_id = pad_id

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8,
                                                   activation="gelu", dropout=dropout,
                                                   dim_feedforward=hidden_size + hidden_size // 2,
                                                   batch_first=True)
        self.finetune_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)

        self.question_attention = nn.Linear(hidden_size, hidden_size)
        self.context_attention = nn.Linear(hidden_size, hidden_size)
        self.za_attention = nn.Linear(nza, hidden_size)

        self.za_linear = nn.Linear(2 * hidden_size, nza * nzadim)
        self.zq_linear = nn.Linear(5 * hidden_size + nza, 2 * nzqdim)

    def forward(self, c_ids, q_ids, a_ids):
        N, _ = c_ids.size()
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        q_mask = return_attention_mask(q_ids, self.pad_token_id)

        """ Answer encoder """
        # context enc
        c_embeddings = self.context_encoder(input_ids=c_ids, attention_mask=c_mask)[0]
        # shape = (N, seq_len, hidden_size) and (N, hidden_size), in order
        c_hs = self.finetune_encoder(c_embeddings, mask=c_mask)
        c_h = c_hs[:, 0] # CLS token

        # context and answer enc
        c_a_ids = c_ids * a_ids
        c_a_ids[:, 0] = c_ids[:, 0] # CLS token is there for capturing context
        c_a_mask = return_attention_mask(c_a_ids, self.pad_token_id)
        c_a_embeddings = self.context_encoder(input_ids=c_a_ids, attention_mask=c_a_mask)[0]
        c_a_hs = self.finetune_encoder(c_a_embeddings, mask=c_a_mask)
        c_a_h = c_a_hs[:, 0] # CLS token

        h = torch.cat([c_h, c_a_h], dim=-1)
        za_logits = self.za_linear(h).view(-1, self.nza, self.nzadim)
        za = gumbel_softmax(za_logits, hard=True)

        """ Question encoder """
        # shape = (N, seq_len, hidden_size)
        q_embeddings = self.context_encoder(input_ids=q_ids, attention_mask=q_mask)[0]
        q_hs = self.finetune_encoder(q_embeddings, mask=q_mask)
        q_h = q_hs[:, 0] # CLS token

        # attention q, c
        # For attention calculation, linear layer is there for projection
        c_attned_by_q, _ = cal_attn(self.question_attention(q_h).unsqueeze(1),
                                    c_hs, c_mask.unsqueeze(1)).squeeze(1)

        # attetion c, q
        # For attention calculation, linear layer is there for projection
        q_attned_by_c, _ = cal_attn(self.context_attention(c_h).unsqueeze(1),
                                    q_hs, q_mask.unsqueeze(1)).squeeze(1)

        # attention za, q
        q_attned_by_za, _ = cal_attn(self.za_attention(softargmax(za)).unsqueeze(1),
                                    q_hs, q_mask.unsqueeze(1))

        h = torch.cat([softargmax(za), q_h, c_h, q_attned_by_c, c_attned_by_q, q_attned_by_za], dim=-1)

        zq_mu, zq_logvar = torch.split(self.zq_linear(h), self.nzqdim, dim=1)
        zq = sample_gaussian(zq_mu, zq_logvar)

        if self.training:
            return zq_mu, zq_logvar, zq, za_logits, za
        else:
            return zq, za
