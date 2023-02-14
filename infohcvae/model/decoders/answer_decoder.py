import torch
import torch.nn as nn
import torch.nn.functional as F
from infohcvae.model.custom.answer_decoder_bilstm_with_attention import AnswerDecoderBiLstmWithAttention
from infohcvae.model.model_utils import (
    return_attention_mask, return_inputs_length
)


class AnswerDecoder(nn.Module):
    def __init__(self, embedding, d_model, nzqdim, nzadim, nza_values,
                 lstm_dec_nhidden, lstm_dec_nlayers, dropout=0.0, pad_token_id=0):
        super(AnswerDecoder, self).__init__()

        self.context_encoder = embedding

        self.pad_token_id = pad_token_id

        self.nzadim = nzadim
        self.nza_values = nza_values
        self.dec_nhidden = lstm_dec_nhidden
        self.za_projection = nn.Linear(nzadim * nza_values, d_model)
        self.zq_projection = nn.Linear(nzqdim, lstm_dec_nhidden * 2)

        layers = [AnswerDecoderBiLstmWithAttention(4 * d_model, lstm_dec_nhidden, 1, dropout=dropout)]
        for i in range(1, lstm_dec_nlayers):
            layers.append(AnswerDecoderBiLstmWithAttention(2 * lstm_dec_nhidden, lstm_dec_nhidden, 1, dropout=dropout))
        self.answer_decoder = nn.ModuleList(layers)

        self.start_linear = nn.Linear(2 * lstm_dec_nhidden, 1)
        self.end_linear = nn.Linear(2 * lstm_dec_nhidden, 1)

    def _build_za_init_state(self, za, max_c_len):
        z_projected = self.za_projection(za.view(-1, self.nzadim * self.nza_values))  # shape = (N, d_model)
        z_projected = z_projected.unsqueeze(1).expand(-1, max_c_len, -1)  # shape = (N, c_len, d_model)
        return z_projected

    def _build_zq_init_state(self, zq):
        q_init = self.zq_projection(zq)  # shape = (N, d_model)
        q_init = q_init.view(-1, 2, self.dec_nhidden).transpose(0, 1).contiguous()
        q_state = (q_init, q_init)
        return q_state

    def forward(self, c_ids, zq, za):
        _, max_c_len = c_ids.size()

        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_lengths = return_inputs_length(c_mask)

        c_embeds = self.context_encoder(c_ids, c_mask)
        init_state = self._build_za_init_state(za, max_c_len)
        q_init_state = self._build_zq_init_state(zq)
        dec_hs = torch.cat([c_embeds, init_state,
                            c_embeds * init_state,
                            torch.abs(c_embeds - init_state)],
                           dim=-1)
        for layer in self.answer_decoder:
            dec_hs = layer(dec_hs, c_lengths, c_mask, q_init_state)

        start_logits = self.start_linear(dec_hs).squeeze(-1)
        end_logits = self.end_linear(dec_hs).squeeze(-1)
        # joint_logits = torch.bmm(dec_hs, dec_hs.transpose(-1, -2))

        start_end_mask = (c_mask == 0)
        masked_start_logits = start_logits.masked_fill(start_end_mask, -3e4)
        masked_end_logits = end_logits.masked_fill(start_end_mask, -3e4)

        # start_end_mask_matrix = torch.matmul(start_end_mask.unsqueeze(2).float(), start_end_mask.unsqueeze(1).float())
        # start_end_mask_matrix = torch.triu(start_end_mask_matrix) == 0
        # masked_joint_logits = joint_logits.masked_fill(start_end_mask_matrix, -3e4)

        return masked_start_logits, masked_end_logits #, masked_joint_logits

    def generate(self, c_ids, zq=None, za=None, start_logits=None, end_logits=None):
        assert (start_logits is None and end_logits is None) or (start_logits is not None and end_logits is not None),\
            "`start_logits` and `end_logits` must be both provided or both empty"
        assert (start_logits is None and (za is not None or zq is not None)) or \
               (start_logits is not None and (za is None or zq is None)), \
            "cannot both provide logits and latent `zq` and `za`, only <one> is accepted"

        if start_logits is None:
            start_logits, end_logits = self.forward(c_ids, zq, za)

        batch_size, max_c_len = c_ids.size()

        c_mask = return_attention_mask(c_ids, self.pad_token_id)

        mask = torch.matmul(c_mask.unsqueeze(2).float(), c_mask.unsqueeze(1).float())
        mask = torch.triu(mask) == 0
        score = (F.log_softmax(start_logits, dim=1).unsqueeze(2)
                 + F.log_softmax(end_logits, dim=1).unsqueeze(1))
        score = score.masked_fill(mask, -3e4)
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

    # def generate_with_joint_logits(self, c_ids, zq=None, za=None, joint_logits=None,
    #                                score="logprobs", has_sentinel=False, max_answer_length=30):
    #     """
    #             This method has been borrowed from AllenNLP
    #             :param valid_span_log_probs:
    #             :return:
    #     """
    #     assert (joint_logits is None and (za is not None or zq is not None)) or \
    #            (joint_logits is not None and (za is None or zq is None)), \
    #         "cannot both provide logits and latent `zq` and `za`, only <one> is accepted"
    #
    #     if joint_logits is None:
    #         _, _, joint_logits = self.forward(c_ids, zq, za)
    #
    #     batch_size, max_c_len = c_ids.size()
    #
    #     # if first token is sentinel, class, combinations (0,x) and (x,0); x!=0 are invalid
    #     # mask these
    #     if has_sentinel:
    #         joint_logits[:, 1:, 0] = -3e4
    #         joint_logits[:, 0, 1:] = -3e4
    #
    #     # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    #     # can recover the start and end indices from this flattened list using simple modular
    #     # arithmetic.
    #     # (batch_size, passage_length * passage_length)
    #     spans_longer_than_maxlen_mask = torch.Tensor(
    #         [[j - i + 1 > max_answer_length for j in range(max_c_len)] for i in range(max_c_len)]) \
    #         .to(joint_logits.get_device() if joint_logits.get_device() >= 0 else torch.device("cpu"))
    #     joint_logits.masked_fill_(spans_longer_than_maxlen_mask.unsqueeze(0).bool(), -3e4)
    #     joint_logits = joint_logits.view(batch_size, -1)
    #     if score == "probs":
    #         scores = F.softmax(joint_logits, dim=-1)
    #     elif score == "logprobs":
    #         scores = joint_logits
    #     else:
    #         raise NotImplemented(f"Unknown score type \"{score}\"")
    #
    #     scores, start_positions = scores.max(dim=1)
    #     scores, end_positions = scores.max(dim=1)
    #     start_positions = torch.gather(start_positions, 1, end_positions.view(-1, 1)).squeeze(1)
    #
    #     idxes = torch.arange(0, max_c_len, out=torch.LongTensor(max_c_len))
    #     idxes = idxes.unsqueeze(0).to(joint_logits.device).expand(batch_size, -1)
    #
    #     start_positions = start_positions.unsqueeze(1)
    #     start_mask = (idxes >= start_positions).long()
    #     end_positions = end_positions.unsqueeze(1)
    #     end_mask = (idxes <= end_positions).long()
    #     answer_mask = start_mask + end_mask - 1
    #
    #     return answer_mask, start_positions.squeeze(1), end_positions.squeeze(1)
