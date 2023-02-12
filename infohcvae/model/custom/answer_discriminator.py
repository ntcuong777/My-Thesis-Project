import torch
import torch.nn as nn
from infohcvae.model.custom.custom_lstm import CustomLSTM
from infohcvae.model.model_utils import (
    return_attention_mask, return_inputs_length
)


class AnswerDiscriminator(nn.Module):
    def __init__(self, embedding, d_model, hidden_size, num_layers, dropout=0.0, pad_token_id=0):
        super(AnswerDiscriminator, self).__init__()

        self.embedding = embedding

        self.pad_token_id = pad_token_id
        self.nhidden = hidden_size
        self.nlayers = num_layers

        self.discriminator = CustomLSTM(input_size=d_model * 2, hidden_size=hidden_size,
                                        num_layers=num_layers, dropout=dropout,
                                        bidirectional=True)
        self.linear = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size, eps=1e-05, momentum=0.1),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1)
        )
        self.linear.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02) # N(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, c_ids, a_mask):
        _, max_c_len = c_ids.size()

        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_lengths = return_inputs_length(c_mask)

        c_embeds = self.embedding(c_ids)
        c_a_embeds = c_embeds * a_mask.unsqueeze(2)
        _, combined_c_a_states = self.discriminator(torch.cat([c_embeds, c_a_embeds], dim=-1), c_lengths.to("cpu"))
        combined_c_a_h = combined_c_a_states[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        combined_c_a_h = combined_c_a_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)
        return self.linear(combined_c_a_h)
