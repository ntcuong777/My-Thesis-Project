import torch
import torch.nn as nn
import torch.nn.functional as F
from infohcvae.model.custom.custom_lstm import CustomLSTM
from infohcvae.model.custom.gated_self_attention import GatedAttention
from infohcvae.model.model_utils import (
    return_attention_mask, return_inputs_length
)


class AnswerDecoder(nn.Module):
    def __init__(self, embedding, d_model, nzadim, nza_values,
                 lstm_dec_nhidden, lstm_dec_nlayers, dropout=0.0, pad_token_id=0):
        super(AnswerDecoder, self).__init__()

        self.context_encoder = embedding

        self.pad_token_id = pad_token_id

        self.nzadim = nzadim
        self.nza_values = nza_values
        self.za_projection = nn.Sequential(
            nn.Linear(nzadim * nza_values, d_model),
            nn.BatchNorm1d(d_model, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model, eps=1e-05, momentum=0.1),
            nn.Tanh(),
            nn.Linear(d_model, d_model, bias=False)
        )

        self.answer_decoder = CustomLSTM(input_size=4 * d_model, hidden_size=lstm_dec_nhidden,
                                         num_layers=lstm_dec_nlayers, dropout=dropout,
                                         bidirectional=True)
        self.self_attention = GatedAttention(2 * lstm_dec_nhidden)

        self.answer_token_discriminator = nn.Sequential(
            nn.Linear(2 * lstm_dec_nhidden, lstm_dec_nhidden),
            nn.ReLU(),
            nn.Linear(lstm_dec_nhidden, lstm_dec_nhidden),
            nn.ReLU(),
            nn.Linear(lstm_dec_nhidden, 1)
        )
        self.answer_token_discriminator.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02) # N(0, 0.02)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def _build_za_init_state(self, za, max_c_len):
        z_projected = self.za_projection(za.view(-1, self.nzadim * self.nza_values))  # shape = (N, d_model)
        z_projected = z_projected.unsqueeze(1).expand(-1, max_c_len, -1)  # shape = (N, c_len, d_model)
        return z_projected

    def forward(self, c_ids, za):
        _, max_c_len = c_ids.size()

        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_lengths = return_inputs_length(c_mask)

        c_embeds = self.context_encoder(c_ids, c_mask)
        init_state = self._build_za_init_state(za, max_c_len)
        dec_inputs = torch.cat([c_embeds, init_state,
                                c_embeds * init_state,
                                torch.abs(c_embeds - init_state)],
                               dim=-1)
        dec_outputs, _ = self.answer_decoder(dec_inputs, c_lengths.to("cpu"))
        dec_outputs = self.self_attention(dec_outputs, c_mask)

        answer_tok_logits = self.answer_token_discriminator(dec_outputs).squeeze(2)

        start_end_mask = (c_mask == 0)
        masked_answer_tok_logits = answer_tok_logits.masked_fill(start_end_mask, -3e4)
        return masked_answer_tok_logits

    def generate(self, c_ids, za=None, answer_tok_logits=None):
        assert (answer_tok_logits is None and za is not None) or (answer_tok_logits is not None and za is None), \
            "cannot both provide logits and latent `za`, only <one> is accepted"

        if answer_tok_logits is None:
            answer_tok_logits = self.forward(c_ids, za)

        batch_size, max_c_len = c_ids.size()

        c_mask = return_attention_mask(c_ids, self.pad_token_id)

        answer_tok_logits = answer_tok_logits.masked_fill(c_mask, -3e4)
        answer_tok_logits = torch.round(answer_tok_logits)
        start_positions = answer_tok_logits.argmax(dim=1)
        end_positions = max_c_len - torch.flip(answer_tok_logits, dims=[1]).max(dim=1) - 1

        idxes = torch.arange(0, max_c_len, out=torch.LongTensor(max_c_len))
        idxes = idxes.unsqueeze(0).to(answer_tok_logits.device).repeat(batch_size, 1)

        start_positions = start_positions.unsqueeze(1)
        start_mask = (idxes >= start_positions).long()
        end_positions = end_positions.unsqueeze(1)
        end_mask = (idxes <= end_positions).long()
        answer_mask = start_mask + end_mask - 1

        return answer_mask, start_positions.squeeze(1), end_positions.squeeze(1)
