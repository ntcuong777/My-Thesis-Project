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
                 lstm_dec_nhidden, lstm_dec_nlayers, dropout=0.0):
        super(AnswerDecoder, self).__init__()

        self.embedding = embedding

        self.nzadim = nzadim
        self.nza_values = nza_values
        self.za_projection = nn.Linear(nzadim * nza_values, d_model, bias=False)

        self.answer_decoder = CustomLSTM(input_size=4 * d_model, hidden_size=lstm_dec_nhidden,
                                         num_layers=lstm_dec_nlayers, dropout=dropout,
                                         bidirectional=True)
        self.self_attention = GatedAttention(2 * lstm_dec_nhidden)

        self.start_linear = nn.Linear(2 * lstm_dec_nhidden, 1)
        self.end_linear = nn.Linear(2 * lstm_dec_nhidden, 1)

    def _build_za_init_state(self, za, max_c_len):
        z_projected = self.za_projection(za.view(-1, self.nzadim * self.nza_values))  # shape = (N, d_model)
        z_projected = z_projected.unsqueeze(1).expand(-1, max_c_len, -1)  # shape = (N, c_len, d_model)
        return z_projected

    def forward(self, c_ids, za, return_embeds=None):
        _, max_c_len = c_ids.size()

        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_lengths = return_inputs_length(c_mask)

        c_embeds = self.embedding(c_ids, c_mask)
        init_state = self._build_za_init_state(za, max_c_len)
        dec_inputs = torch.cat([c_embeds, init_state,
                                c_embeds * init_state,
                                torch.abs(c_embeds - init_state)],
                               dim=-1)
        dec_outputs, _ = self.answer_decoder(dec_inputs, c_lengths.to("cpu"))
        dec_outputs = self.self_attention(dec_outputs, c_mask)

        start_logits = self.start_linear(dec_outputs).squeeze(-1)
        end_logits = self.end_linear(dec_outputs).squeeze(-1)

        start_end_mask = (c_mask == 0)
        masked_start_logits = start_logits.masked_fill(start_end_mask, -1e9)
        masked_end_logits = end_logits.masked_fill(start_end_mask, -1e9)

        if return_embeds is not None and return_embeds:
            return masked_start_logits, masked_end_logits, dec_outputs

        return masked_start_logits, masked_end_logits

    def generate(self, c_ids, za=None, start_logits=None, end_logits=None):
        assert (start_logits is None and end_logits is None) or (start_logits is not None and end_logits is not None),\
            "`start_logits` and `end_logits` must be both provided or both empty"
        assert (start_logits is None and za is not None) or (start_logits is not None and za is None), \
            "cannot both provide logits and latent `za`, only <one> is accepted"

        if start_logits is None:
            start_logits, end_logits = self.forward(c_ids, za)

        batch_size, max_c_len = c_ids.size()

        c_mask = return_attention_mask(c_ids, self.pad_token_id)

        mask = torch.matmul(c_mask.unsqueeze(2).float(), c_mask.unsqueeze(1).float())
        mask = torch.triu(mask) == 0
        score = (F.log_softmax(start_logits, dim=1).unsqueeze(2)
                 + F.log_softmax(end_logits, dim=1).unsqueeze(1))
        score = score.masked_fill(mask, -10000.0)
        score, start_positions = score.max(dim=1)
        score, end_positions = score.max(dim=1)
        start_positions = torch.gather(start_positions, 1, end_positions.view(-1, 1)).squeeze(1)

        idxes = torch.arange(0, max_c_len, out=torch.LongTensor(max_c_len))
        idxes = idxes.unsqueeze(0).to(start_logits.device).expand(batch_size, -1)

        start_positions = start_positions.unsqueeze(1)
        start_mask = (idxes >= start_positions).long()
        end_positions = end_positions.unsqueeze(1)
        end_mask = (idxes <= end_positions).long()
        answer_mask = start_mask + end_mask - 1

        return answer_mask, start_positions.squeeze(1), end_positions.squeeze(1)