import numpy as np
import torch
import torch.nn as nn
from infohcvae.model.custom.custom_lstm import CustomLSTM
from infohcvae.model.custom.luong_attention import LuongAttention
from infohcvae.model.custom.custom_context_encoder import CustomContextEncoderForQG
from torch_scatter import scatter_max
from infohcvae.model.model_utils import (
    return_inputs_length, return_attention_mask,
)


class QuestionDecoder(nn.Module):
    def __init__(self, word_embeddings, context_embedding, nzqdim,
                 d_model, lstm_dec_nhidden, lstm_dec_nlayers, sos_id, eos_id,
                 vocab_size, dropout=0.0, max_q_len=64, pad_token_id=0):
        super(QuestionDecoder, self).__init__()

        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_token_id = pad_token_id
        self.d_model = d_model
        self.embedding = word_embeddings
        self.dec_nhidden = lstm_dec_nhidden
        self.vocab_size = vocab_size
        self.dec_nlayers = lstm_dec_nlayers
        # this max_len include sos eos
        self.max_q_len = max_q_len

        self.context_encoder = CustomContextEncoderForQG(
            context_embedding, d_model, lstm_dec_nhidden // 2, lstm_dec_nlayers,
            dropout=dropout, pad_token_id=pad_token_id)

        self.question_lstm = CustomLSTM(input_size=d_model, hidden_size=lstm_dec_nhidden,
                                        num_layers=lstm_dec_nlayers, dropout=dropout,
                                        bidirectional=False)

        self.q_init_hidden_linear = nn.Linear(nzqdim, lstm_dec_nlayers * lstm_dec_nhidden)
        self.q_init_cell_linear = nn.Linear(nzqdim, lstm_dec_nlayers * lstm_dec_nhidden)

        self.question_attention = LuongAttention(lstm_dec_nhidden, lstm_dec_nhidden)

        self.concat_linear = nn.Sequential(nn.Linear(2 * lstm_dec_nhidden, 2 * lstm_dec_nhidden),
                                           nn.Dropout(dropout),
                                           nn.Linear(2 * lstm_dec_nhidden, 2 * d_model))

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # fix output word matrix
        self.lm_head.weight = word_embeddings.word_embeddings.weight
        for param in self.lm_head.parameters():
            param.requires_grad = False

        self.discriminator = nn.Bilinear(d_model, lstm_dec_nhidden, 1)

    def _build_zq_init_state(self, zq):
        q_init_h = self.q_init_hidden_linear(zq)
        q_init_c = self.q_init_cell_linear(zq)
        q_init_h = q_init_h.view(-1, self.dec_nlayers, self.dec_nhidden).transpose(0, 1).contiguous()
        q_init_c = q_init_c.view(-1, self.dec_nlayers, self.dec_nhidden).transpose(0, 1).contiguous()
        q_init_state = (q_init_h, q_init_c)
        return q_init_state

    def forward(self, c_ids, q_ids, c_a_mask, zq, return_qa_mean_embeds=None):
        c_outputs = self.context_encoder(c_ids, c_a_mask)

        c_mask = return_attention_mask(c_ids, self.pad_token_id)

        q_mask = return_attention_mask(q_ids, self.pad_token_id)
        q_lengths = return_inputs_length(q_mask)

        init_state = self._build_zq_init_state(zq)

        # question dec
        q_embeds = self.embedding(q_ids)
        q_outputs, _ = self.question_lstm(q_embeds, q_lengths.to("cpu"), init_state)

        logits, q_last_outputs = self.get_question_logits_from_out_hidden_states(
            c_ids, c_mask, q_ids, q_mask, q_outputs, c_outputs)

        if return_qa_mean_embeds is not None and return_qa_mean_embeds:
            a_emb = c_outputs * c_a_mask.float().unsqueeze(2)
            a_mean_emb = torch.sum(a_emb, dim=1) / c_a_mask.sum(1).unsqueeze(1).float()

            q_emb = q_last_outputs * q_mask.unsqueeze(2)
            q_mean_emb = torch.sum(q_emb, dim=1) / q_lengths.unsqueeze(1).float()
            return logits, (q_mean_emb, a_mean_emb)

        return logits

    def get_question_logits_from_out_hidden_states(
            self, c_ids, c_mask, q_ids, q_mask, q_hidden_states, c_hidden_states):
        batch_size, max_q_len = q_ids.size()

        # context-question attention
        mask = torch.matmul(q_mask.unsqueeze(2), c_mask.unsqueeze(1))
        c_attned_by_q, attn_logits = self.question_attention(
            q_hidden_states, c_hidden_states, mask, return_attention_logits=True)

        # gen logits
        q_concated = torch.cat([q_hidden_states, c_attned_by_q], dim=2)
        q_concated = self.concat_linear(q_concated)
        q_maxouted, _ = q_concated.view(batch_size, max_q_len, self.d_model, 2).max(dim=-1)
        gen_logits = self.lm_head(q_maxouted)

        # copy logits
        bq = batch_size * max_q_len
        context_ids = c_ids.unsqueeze(1).repeat(
            1, max_q_len, 1).view(bq, -1).contiguous()
        attn_logits = attn_logits.view(bq, -1).contiguous()
        copy_logits = torch.zeros(bq, self.vocab_size).to(context_ids.device) - 3e4
        copy_logits, _ = scatter_max(attn_logits, context_ids, out=copy_logits)
        copy_logits = copy_logits.masked_fill(copy_logits == -3e4, 0)
        copy_logits = copy_logits.view(batch_size, max_q_len, -1).contiguous()

        total_logits = gen_logits + copy_logits
        return total_logits, q_maxouted

    def generate(self, c_ids, c_a_mask, zq):
        """ Generate question using greedy decoding """
        def postprocess(q_token_ids):
            eos_mask = q_token_ids == self.eos_id
            no_eos_idx_sum = (eos_mask.sum(dim=1) == 0).long() * \
                             (self.max_q_len - 1)
            eos_mask = eos_mask.cpu().numpy()
            q_lengths = np.argmax(eos_mask, axis=1) + 1
            q_lengths = torch.tensor(q_lengths).to(
                q_token_ids.device).long() + no_eos_idx_sum
            batch_size, max_len = q_token_ids.size()
            idxes = torch.arange(0, max_len).to(q_token_ids.device)
            idxes = idxes.unsqueeze(0).repeat(batch_size, 1)
            q_mask = (idxes < q_lengths.unsqueeze(1))
            q_token_ids = q_token_ids.long() * q_mask.long()
            return q_token_ids

        c_mask = return_attention_mask(c_ids, self.pad_token_id)

        c_outputs = self.context_encoder(c_ids, c_a_mask)

        batch_size = c_ids.size(0)

        q_ids = torch.LongTensor([self.sos_id] * batch_size).unsqueeze(1)
        q_ids = q_ids.to(c_ids.device)
        q_lengths = torch.ones_like(q_ids).squeeze(1)
        token_type_ids = torch.zeros_like(q_ids)
        position_ids = torch.zeros_like(q_ids)
        q_embeddings = self.embedding(input_ids=q_ids, token_type_ids=token_type_ids, position_ids=position_ids)

        state = self._build_zq_init_state(zq)

        # unroll
        all_q_ids = list()
        all_q_ids.append(q_ids)
        for _ in range(self.max_q_len - 1):
            position_ids = position_ids + 1
            q_outputs, state = self.question_lstm(q_embeddings, q_lengths.to("cpu"), state)

            logits, _ = self.get_question_logits_from_out_hidden_states(
                c_ids, c_mask, q_ids, torch.ones_like(q_ids, dtype=torch.float), q_outputs, c_outputs)

            q_ids = torch.argmax(logits, 2)
            all_q_ids.append(q_ids)

            q_embeddings = self.embedding(input_ids=q_ids, token_type_ids=token_type_ids, position_ids=position_ids)
            q_lengths = torch.ones_like(q_ids).squeeze(1)

        q_ids = torch.cat(all_q_ids, 1)
        q_ids = postprocess(q_ids)

        return q_ids
