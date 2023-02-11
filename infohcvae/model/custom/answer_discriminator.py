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

        self.discriminator = CustomLSTM(input_size=d_model, hidden_size=hidden_size,
                                        num_layers=num_layers, dropout=dropout,
                                        bidirectional=True)
        self.linear = nn.Linear(2 * hidden_size, 1)
        self.init_weights(self.linear)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02) # N(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, c_ids, a_mask):
        _, max_c_len = c_ids.size()

        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_lengths = return_inputs_length(c_mask)

        c_embeds = self.embedding(c_ids, a_mask)
        _, c_states = self.discriminator(c_embeds, c_lengths.to("cpu"))
        c_h = c_states[0].view(self.nlayers, 2, -1, self.nhidden)[-1]
        c_h = c_h.transpose(0, 1).contiguous().view(-1, 2 * self.nhidden)
        return self.linear(c_h)
