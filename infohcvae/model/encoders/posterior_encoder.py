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
                                                   dim_feedforward=hidden_size,
                                                   batch_first=True)
        self.finetune_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_enc_layers)

        self.za_attention = nn.Linear(nza * nzadim, hidden_size)

        self.za_linear = nn.Linear(4 * hidden_size, nza * nzadim)
        self.zq_linear = nn.Linear(5 * hidden_size + nza * nzadim, 2 * nzqdim)

    def forward(self, c_ids, q_ids, a_ids):
        N, _ = c_ids.size()
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        q_mask = return_attention_mask(q_ids, self.pad_token_id)

        """ Answer encoder """
        # context enc
        c_embeddings = self.context_encoder(input_ids=c_ids, attention_mask=c_mask)[0]
        # shape = (N, seq_len, hidden_size) and (N, hidden_size), in order
        c_hs = self.finetune_encoder(c_embeddings, src_key_padding_mask=c_mask)
        c_h = c_hs[:, 0] # CLS token

        # context and answer enc
        c_a_embeddings = self.context_encoder(input_ids=c_ids, attention_mask=c_mask, token_type_ids=a_ids)[0]
        c_a_hs = self.finetune_encoder(c_a_embeddings, src_key_padding_mask=c_mask)
        c_a_h = c_a_hs[:, 0] # CLS token

        mask_out_a_ids = c_ids * a_ids
        mask_out_a_ids[:, 0] = c_ids[:, 0] # still keep CLS token
        a_ids_mask = return_attention_mask(mask_out_a_ids, self.pad_token_id)
        a_embeddings = self.context_encoder(input_ids=mask_out_a_ids, attention_mask=a_ids_mask)[0]
        a_hs = self.finetune_encoder(a_embeddings, src_key_padding_mask=a_ids_mask)
        a_h = a_hs[:, 0] # CLS token

        # attention a, c
        # For attention calculation, linear layer is there for projection
        c_attned_by_a = cal_attn(a_h.unsqueeze(1),
                                c_hs, c_mask.unsqueeze(1))[0].squeeze(1)

        # attention c, a
        # For attention calculation, linear layer is there for projection
        a_attned_by_c = cal_attn(c_h.unsqueeze(1),
                                a_hs, c_mask.unsqueeze(1))[0].squeeze(1)

        h = torch.cat([a_h, c_a_h, c_attned_by_a, a_attned_by_c], dim=-1)
        za_logits = self.za_linear(h).view(-1, self.nza, self.nzadim)
        za = gumbel_softmax(za_logits, hard=True)

        """ Question encoder """
        # shape = (N, seq_len, hidden_size)
        q_embeddings = self.context_encoder(input_ids=q_ids, attention_mask=q_mask)[0]
        q_hs = self.finetune_encoder(q_embeddings, src_key_padding_mask=q_mask)
        q_h = q_hs[:, 0]  # CLS token

        # attention q, c
        # For attention calculation, linear layer is there for projection
        c_attned_by_q = cal_attn(q_h.unsqueeze(1),
                                c_hs, c_mask.unsqueeze(1))[0].squeeze(1)

        # attetion c, q
        # For attention calculation, linear layer is there for projection
        q_attned_by_c = cal_attn(c_h.unsqueeze(1),
                                q_hs, q_mask.unsqueeze(1))[0].squeeze(1)

        # attention za, q
        q_attned_by_za = cal_attn(self.za_attention(za.view(-1, self.nza*self.nzadim)).unsqueeze(1),
                                    q_hs, q_mask.unsqueeze(1))[0].squeeze(1)

        h = torch.cat([q_h, c_h, q_attned_by_c, c_attned_by_q, q_attned_by_za, za], dim=-1)

        zq_mu, zq_logvar = torch.split(self.zq_linear(h), self.nzqdim, dim=1)
        zq = sample_gaussian(zq_mu, zq_logvar)

        if self.training:
            return zq_mu, zq_logvar, zq, za_logits, za
        else:
            return zq, za
