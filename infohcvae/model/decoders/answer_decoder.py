import torch
import torch.nn as nn
from infohcvae.model.model_utils import return_attention_mask, softargmax

# TODO: major work to do here
class AnswerDecoder(nn.Module):
    def __init__(self, pad_id, context_enc, hidden_size, nzadim, nza_values, n_dec_layers, dropout=0.0):
        super(AnswerDecoder, self).__init__()

        self.context_encoder = context_enc
        self.pad_token_id = pad_id
        # self.nza = nza
        self.nzadim = nzadim
        self.nza_values = nza_values
        self.hidden_size = hidden_size

        self.za_linear = nn.Linear(nzadim * nza_values, hidden_size)

        self.start_linear = nn.Linear(hidden_size, 1)
        self.end_linear = nn.Linear(hidden_size, 1)
        self.ls = nn.LogSoftmax(dim=1)

    def forward(self, c_ids, za):
        """
            c_ids: shape = (N, seq_len)
            za: shape = (N, nza, nzadim) where nza is the latent dim,
                nzadim is the categorical dim
        """
        _, max_c_len = c_ids.size()
        c_mask = return_attention_mask(c_ids, self.pad_token_id)

        decoded_a = self.za_linear(za)  # shape = (N, hidden_size)

        # context enc
        # shape = (N, seq_len, hidden_size)
        c_embeddings = self.context_encoder(input_ids=c_ids, attention_mask=c_mask)[0]

        repeated_decoded_a = decoded_a.unsqueeze(1).repeat(1, max_c_len, 1)
        h = torch.cat([c_embeddings, repeated_decoded_a], dim=-1)
        out_features = self.answer_decoder(h, src_key_padding_mask=c_mask)

        start_logits = self.start_linear(out_features).squeeze(-1)
        end_logits = self.end_linear(out_features).squeeze(-1)

        start_end_mask = (c_mask == 0)
        masked_start_logits = start_logits.masked_fill(
            start_end_mask, -10000.0)
        masked_end_logits = end_logits.masked_fill(start_end_mask, -10000.0)

        return masked_start_logits, masked_end_logits

    def generate(self, c_ids, za):
        start_logits, end_logits = self.forward(c_ids, za)
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        batch_size, max_c_len = c_ids.size()

        mask = torch.matmul(c_mask.unsqueeze(2).float(),
                            c_mask.unsqueeze(1).float())
        mask = torch.triu(mask) == 0
        score = (self.ls(start_logits).unsqueeze(2)
                 + self.ls(end_logits).unsqueeze(1))
        score = score.masked_fill(mask, -10000.0)
        score, start_positions = score.max(dim=1)
        score, end_positions = score.max(dim=1)
        start_positions = torch.gather(start_positions,
                                       1,
                                       end_positions.view(-1, 1)).squeeze(1)

        idxes = torch.arange(0, max_c_len, out=torch.LongTensor(max_c_len))
        idxes = idxes.unsqueeze(0).to(
            start_logits.device).repeat(batch_size, 1)

        start_positions = start_positions.unsqueeze(1)
        start_mask = (idxes >= start_positions).long()
        end_positions = end_positions.unsqueeze(1)
        end_mask = (idxes <= end_positions).long()
        a_ids = start_mask + end_mask - 1

        return a_ids, start_positions.squeeze(1), end_positions.squeeze(1)
