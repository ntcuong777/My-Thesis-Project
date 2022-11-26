import torch
import torch.nn as nn
from model.customized_layers import CustomLSTM
from model.model_utils import return_mask_lengths, cal_attn, sample_gaussian

class PriorEncoder(nn.Module):
    def __init__(self, embedding, emsize,
                 nhidden, nlayers,
                 nzqdim, nzadim,
                 dropout=0):
        super(PriorEncoder, self).__init__()

        self.embedding = embedding
        self.nhidden = nhidden
        self.nlayers = nlayers
        self.nzqdim = nzqdim
        # self.nza = nza
        self.nzadim = nzadim

        self.context_encoder = CustomLSTM(input_size=emsize,
                                          hidden_size=nhidden,
                                          num_layers=nlayers,
                                          dropout=dropout,
                                          bidirectional=True)

        self.zq_attention = nn.Linear(nzqdim, 2 * nhidden)

        self.zq_linear = nn.Linear(2 * nhidden, 2 * nzqdim)
        self.za_linear = nn.Linear(nzqdim + 2 * 2 * nhidden, 2 * nzadim)


    def forward(self, c_ids):
        c_mask, c_lengths = return_mask_lengths(c_ids)

        c_embeddings = self.embedding(c_ids)
        c_hs, c_state = self.context_encoder(c_embeddings, c_lengths.to("cpu"))
        c_h = c_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_h = c_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        zq_mu, zq_logvar = torch.split(self.zq_linear(c_h), self.nzqdim, dim=1)
        zq = sample_gaussian(zq_mu, zq_logvar)

        mask = c_mask.unsqueeze(1)
        c_attned_by_zq, _ = cal_attn(self.zq_attention(zq).unsqueeze(1),
                                     c_hs,
                                     mask)
        c_attned_by_zq = c_attned_by_zq.squeeze(1)

        h = torch.cat([zq, c_attned_by_zq, c_h], dim=-1)

        # za_logits = self.za_linear(h).view(-1, self.nza, self.nzadim)
        # # za_prob = F.softmax(za_logits, dim=-1)
        # za = gumbel_softmax(za_logits, hard=True)
        za_mu, za_logvar = torch.split(self.za_linear(h), self.nzadim, dim=1)
        za = sample_gaussian(za_mu, za_logvar)

        if self.training:
            return zq_mu, zq_logvar, zq, za_mu, za_logvar, za
        else:
            return zq, za


    def interpolation(self, c_ids, zq):
        c_mask, c_lengths = return_mask_lengths(c_ids)

        c_embeddings = self.embedding(c_ids)
        c_hs, c_state = self.context_encoder(c_embeddings, c_lengths.to("cpu"))
        c_h = c_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_h = c_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        mask = c_mask.unsqueeze(1)
        c_attned_by_zq, _ = cal_attn(
            self.zq_attention(zq).unsqueeze(1), c_hs, mask)
        c_attned_by_zq = c_attned_by_zq.squeeze(1)

        h = torch.cat([zq, c_attned_by_zq, c_h], dim=-1)

        # za_logits = self.za_linear(h).view(-1, self.nza, self.nzadim)
        # za = gumbel_softmax(za_logits, hard=True)
        za_mu, za_logvar = torch.split(self.za_linear(h), self.nzadim, dim=1)
        za = sample_gaussian(za_mu, za_logvar)

        return za