import torch
import torch.nn as nn
from infohcvae.model.model_utils import return_attention_mask, \
    cal_attn, gumbel_softmax, sample_gaussian, softargmax


class PosteriorEncoder(nn.Module):
    def __init__(self, pad_id, context_enc, hidden_size,
                 latent_dim, nvalues, max_len, dropout=0.0):
        super(PosteriorEncoder, self).__init__()

        self.context_encoder = context_enc
        self.hidden_size = hidden_size
        # self.nza = nza
        self.latent_dim = latent_dim
        self.nvalues = nvalues
        self.pad_token_id = pad_id

        self.z_linear = nn.Linear(max_len * hidden_size, latent_dim * nvalues)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, a_ids):
        N, _ = input_ids.size()
        input_mask = return_attention_mask(input_ids, self.pad_token_id)

        """ Question & Answer encoder """
        # context enc
        input_embeddings = self.context_encoder(input_ids=input_ids, attention_mask=input_mask,
                                                token_type_ids=a_ids)[0]
        # c_h = c_embeddings[:, 0] # CLS token

        z_logits = self.za_linear(input_embeddings).view(-1, self.latent_dim, self.nvalues)
        z = gumbel_softmax(z_logits, hard=True)

        if self.training:
            return z_logits, z
        else:
            return z
