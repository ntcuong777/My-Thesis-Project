import torch
import torch.nn as nn
from infohcvae.model.model_utils import return_attention_mask
from transformers import T5Config
from infohcvae.model.custom import T5ModelAnswerDecoder

class AnswerDecoder(nn.Module):
    def __init__(self, pad_id, nzadim, nza_values,
                 n_dec_finetune_layers=2, base_model="t5-base"):
        super(AnswerDecoder, self).__init__()

        config = T5Config.from_pretrained(base_model)
        self.config = config
        self.t5_model_with_removed_encoder = T5ModelAnswerDecoder.from_pretrained(base_model,
                                                                                  nzadim=nzadim, nza_values=nza_values)

        # Freeze all T5 layers
        for param in self.t5_model_with_removed_encoder.parameters():
            param.requires_grad = False
        # Only some top layers of the decoder are for fine-tuning
        for idx in range(config.num_layers - n_dec_finetune_layers, config.num_layers):
            for param in self.t5_model_with_removed_encoder.get_decoder().block[idx].parameters():
                param.requires_grad = True

        self.pad_token_id = pad_id
        self.nzadim = nzadim
        self.nza_values = nza_values

        self.start_linear = nn.Linear(config.d_model, 1)
        self.end_linear = nn.Linear(config.d_model, 1)
        self.ls = nn.LogSoftmax(dim=1)

    def forward(self, c_ids, za):
        """
            c_ids: shape = (N, seq_len)
            za: shape = (N, nza, nzadim) where nza is the latent dim,
                nzadim is the categorical dim
        """
        _, max_c_len = c_ids.size()
        c_mask = return_attention_mask(c_ids, self.pad_token_id)

        out_features = self.t5_model_with_removed_encoder(context_ids=c_ids, context_mask=c_mask,
                                                          sampled_za=za)[0]

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
        a_mask = start_mask + end_mask - 1

        return a_mask, start_positions.squeeze(1), end_positions.squeeze(1)
