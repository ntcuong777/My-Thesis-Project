import torch
import torch.nn as nn
from model.customized_layers import CustomLSTM
from model.model_utils import return_mask_lengths, cal_attn, sample_gumbel

class DecodingInitStateEncoder(nn.Module):
    def __init__(self, embedding, emsize,
                 nhidden, nlayers, dec_q_nlayers, dec_q_nhidden,
                 nzqdim, nza, nzadim,
                 dropout=0):
        super(DecodingInitStateEncoder, self).__init__()

        self.embedding = embedding
        self.nhidden = nhidden
        self.nlayers = nlayers
        self.dec_q_nlayers = dec_q_nlayers
        self.dec_q_nhidden = dec_q_nhidden
        self.nzqdim = nzqdim
        self.nza = nza
        self.nzadim = nzadim

        self.context_encoder = CustomLSTM(input_size=emsize,
                                          hidden_size=nhidden,
                                          num_layers=nlayers,
                                          dropout=dropout,
                                          bidirectional=True)

        self.zq_attention = nn.Linear(nzqdim, 2 * nhidden)

        self.q_c_init_state_linear = nn.Linear(2 * nhidden + nzqdim, dec_q_nlayers * dec_q_nhidden)
        self.q_h_init_state_linear = nn.Linear(2 * nhidden + nzqdim, dec_q_nlayers * dec_q_nhidden)
        # self.a_init_state_linear = nn.Linear(nzqdim + 2 * 2 * nhidden + nza*nzadim, emsize)
        self.a_init_state_linear = nn.Linear(nzqdim + 2 * 2 * nhidden + nzadim, emsize)


    def forward(self, c_ids):
        c_mask, c_lengths = return_mask_lengths(c_ids)

        prior_zq = torch.randn(c_ids.size(0), self.nzqdim).to(c_ids.device)

        c_embeddings = self.embedding(c_ids)
        c_hs, c_state = self.context_encoder(c_embeddings, c_lengths.to("cpu"))
        c_h = c_state[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_h = c_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)

        q_init_c = self.q_c_init_state_linear(torch.cat((c_h, prior_zq), dim=-1))
        q_init_h = self.q_h_init_state_linear(torch.cat((c_h, prior_zq), dim=-1))
        q_init_h = q_init_h.view(-1, self.dec_q_nlayers, self.dec_q_nhidden).transpose(0, 1).contiguous()
        q_init_c = q_init_c.view(-1, self.dec_q_nlayers, self.dec_q_nhidden).transpose(0, 1).contiguous()
        q_init_state = (q_init_h, q_init_c)

        # For attention calculation, linear layer is there for projection
        mask = c_mask.unsqueeze(1)
        c_attned_by_zq, _ = cal_attn(self.zq_attention(prior_zq).unsqueeze(1),
                                     c_hs,
                                     mask)
        c_attned_by_zq = c_attned_by_zq.squeeze(1)

        h = torch.cat([prior_zq, c_attned_by_zq, c_h], dim=-1)

        # prior_za = sample_gumbel((c_ids.size(0), self.nza, self.nzadim), prior_zq.device)
        prior_za = torch.randn(c_ids.size(0), self.nzadim).to(c_ids.device)
        # a_init_state = self.a_init_state_linear(torch.cat((h, prior_za.view(-1, self.nza*self.nzadim)), dim=-1))
        a_init_state = self.a_init_state_linear(torch.cat((h, prior_za), dim=-1))

        return q_init_state, a_init_state