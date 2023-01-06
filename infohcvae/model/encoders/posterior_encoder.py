import torch
import torch.nn as nn
from infohcvae.model.model_utils import return_attention_mask, \
    cal_attn, gumbel_softmax, sample_gaussian, softargmax


class PosteriorEncoder(nn.Module):
    def __init__(self, pad_id, context_enc, hidden_size,
                 nzqdim, nzadim, nza_values, max_len, dropout=0.0):
        super(PosteriorEncoder, self).__init__()

        self.context_encoder = context_enc
        self.hidden_size = hidden_size
        self.nzqdim = nzqdim
        self.nzadim = nzadim
        self.nza_values = nza_values
        self.pad_token_id = pad_id

        self.zq_linear = nn.Linear(max_len * hidden_size, nzqdim)
        self.za_linear = nn.Linear(max_len * hidden_size, nzadim * nza_values)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, a_ids):
        N, _ = input_ids.size()
        input_mask = return_attention_mask(input_ids, self.pad_token_id)

        """ Question & Answer encoder """
        # context enc
        input_embeddings = self.context_encoder(input_ids=input_ids, attention_mask=input_mask,
                                                token_type_ids=a_ids)[0]
        # input_h = input_embeddings[:, 0] # CLS token
        # can try mean-, max-pooling, or some other pooling scheme

        zq_mu, zq_logvar = torch.split(self.zq_linear(self.dropout(input_embeddings)), self.nzqdim, dim=1)
        zq = sample_gaussian(zq_mu, zq_logvar)

        za_logits = self.za_linear(self.dropout(input_embeddings)).view(-1, self.nzadim, self.nza_values)
        za = gumbel_softmax(za_logits, hard=True)

        if self.training:
            return zq_mu, zq_logvar, zq, za_logits, za
        else:
            return zq, za
