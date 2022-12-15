import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from infohcvae.model.model_utils import return_attention_mask, cal_attn, scatter_max
from infohcvae.model.infomax.dim_bce_infomax import DimBceInfoMax

class _ContextEncoderforQG(nn.Module):
    def __init__(self, pad_id, context_enc, hidden_size, nlayers, dropout=0.2):
        super(_ContextEncoderforQG, self).__init__()
        self.context_encoder = context_enc
        self.pad_token_id = pad_id

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8,
                                                   activation="gelu", dropout=dropout,
                                                   dim_feedforward=hidden_size,
                                                   batch_first=True)
        self.finetune_encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.fusion = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.gate = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    def forward(self, c_ids, a_ids):
        c_mask = return_attention_mask(c_ids, self.pad_token_id)

        c_embeddings = self.context_encoder(input_ids=c_ids, attention_mask=c_mask, token_type_ids=a_ids)[0]
        c_outputs = self.finetune_encoder(c_embeddings, src_key_padding_mask=c_mask)

        c_a_ids = c_ids * a_ids
        c_a_ids[:, 0] = c_ids[:, 0] # add CLS token
        c_a_mask = return_attention_mask(c_a_ids, self.pad_token_id)
        c_a_embeddings = self.context_encoder(input_ids=c_a_ids, attention_mask=c_a_mask)[0]
        c_a_outputs = self.finetune_encoder(c_a_embeddings, src_key_padding_mask=c_a_mask)

        c_concat = torch.cat([c_outputs, c_a_outputs], dim=2)
        c_fused = self.fusion(c_concat).tanh()
        c_gate = self.gate(c_concat).sigmoid()
        c_outputs = c_gate * c_fused + (1 - c_gate) * c_outputs
        return c_outputs


class QuestionDecoder(nn.Module):
    def __init__(self, sos_id, eos_id, pad_id, context_enc, nzqdim,
                 hidden_size, ntokens, n_dec_layers,
                 dropout=0.2, max_q_len=64):
        super(QuestionDecoder, self).__init__()

        self.sos_id = sos_id
        self.eos_id = eos_id
        self.context_encoder = context_enc
        self.hidden_size = hidden_size
        self.n_dec_layers = n_dec_layers
        self.ntokens = ntokens
        # this max_len include sos eos
        self.max_q_len = max_q_len
        self.pad_token_id = pad_id

        self.context_enc_finetuned = _ContextEncoderforQG(pad_id, context_enc, hidden_size, n_dec_layers, dropout)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size*2, nhead=8,
                                                   dim_feedforward=2*hidden_size,
                                                   dropout=dropout, activation="gelu",
                                                   batch_first=True)
        self.question_enc_finetuned = nn.TransformerEncoder(encoder_layer, num_layers=n_dec_layers)
        self.question_enc_linear = nn.Linear(2*hidden_size, hidden_size, bias=False)

        self.question_linear = nn.Linear(hidden_size, hidden_size)

        self.zq_decoder = nn.Linear(nzqdim, hidden_size)

        self.concat_linear = nn.Sequential(nn.Linear(2*hidden_size, 2*hidden_size),
                                           nn.Mish(True),
                                           nn.Dropout(dropout),
                                           nn.Linear(2*hidden_size, 2*hidden_size),
                                           nn.Mish(True))

        self.pre_logit_linear = nn.Linear(hidden_size, context_enc.get_input_embeddings().weight.size(1))
        self.logit_linear = nn.Linear(context_enc.get_input_embeddings().weight.size(0), ntokens, bias=False)

        # fix output word matrix
        self.logit_linear.weight = context_enc.get_input_embeddings().weight
        for param in self.logit_linear.parameters():
            param.requires_grad = False

        self.infomax_est = DimBceInfoMax(hidden_size, hidden_size, use_billinear=True)

    def postprocess(self, q_ids):
        eos_mask = (q_ids == self.eos_id).float()
        no_eos_idx_sum = (eos_mask.sum(dim=1) == 0).long() * \
            (self.max_q_len - 1)
        eos_mask = eos_mask.cpu().numpy()
        q_lengths = np.argmax(eos_mask, axis=1) + 1
        q_lengths = torch.tensor(q_lengths).to(
            q_ids.device).long() + no_eos_idx_sum
        batch_size, max_len = q_ids.size()
        idxes = torch.arange(0, max_len).to(q_ids.device)
        idxes = idxes.unsqueeze(0).repeat(batch_size, 1)
        q_mask = (idxes < q_lengths.unsqueeze(1))
        q_ids = q_ids.long() * q_mask.long()
        return q_ids

    def forward(self, c_ids, q_ids, a_ids, zq):
        batch_size, max_q_len = q_ids.size()

        c_outputs = self.context_enc_finetuned(c_ids, a_ids)

        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        q_mask = return_attention_mask(q_ids, self.pad_token_id)

        decoded_q = self.zq_decoder(zq)  # shape = (N, hidden_size)
        repeated_decoded_q = decoded_q.unsqueeze(1).repeat(1, max_q_len, 1)

        # question dec
        q_embeddings = self.context_encoder(input_ids=q_ids, attention_mask=q_mask)[0]
        q_outputs = self.question_enc_finetuned(torch.cat((q_embeddings, repeated_decoded_q), dim=-1),
                                                src_key_padding_mask=q_mask)
        q_outputs = self.question_enc_linear(q_outputs)

        # attention
        # For attention calculation, linear layer is there for projection
        mask = torch.matmul(q_mask.unsqueeze(2), c_mask.unsqueeze(1))
        c_attned_by_q, attn_logits = cal_attn(self.question_linear(q_outputs),
                                              c_outputs, mask)

        # gen logits
        q_concated = torch.cat([q_outputs, c_attned_by_q], dim=2)
        q_concated = self.concat_linear(q_concated)
        q_maxouted, _ = q_concated.view(
            batch_size, max_q_len, self.hidden_size, 2).max(dim=-1)
        gen_logits = self.logit_linear(self.pre_logit_linear(q_maxouted))

        # copy logits
        bq = batch_size * max_q_len
        c_ids = c_ids.unsqueeze(1).repeat(
            1, max_q_len, 1).view(bq, -1).contiguous()
        attn_logits = attn_logits.view(bq, -1).contiguous()
        copy_logits = torch.zeros(bq, self.ntokens).to(c_ids.device)
        copy_logits = copy_logits - 10000.0
        copy_logits, _ = scatter_max(attn_logits, c_ids, out=copy_logits)
        copy_logits = copy_logits.masked_fill(copy_logits == -10000.0, 0)
        copy_logits = copy_logits.view(batch_size, max_q_len, -1).contiguous()

        logits = gen_logits + copy_logits

        if self.training:
            # mutual information btw answer and question (customized: use bi-lstm to average the question & answer)
            a_emb = c_outputs * a_ids.float().unsqueeze(2)
            a_mean_emb = torch.sum(a_emb, dim=1) / a_ids.sum(1).unsqueeze(1).float()

            q_emb = q_maxouted * q_mask.unsqueeze(2)
            q_mean_emb = torch.sum(q_emb, dim=1) / q_mask.sum(dim=1, keepdim=True).float()

            return logits, self.infomax_est(q_mean_emb, a_mean_emb)
        else:
            return logits

    def generate(self, c_ids, a_ids, zq):
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_outputs = self.context_enc_finetuned(c_ids, a_ids)

        batch_size = c_ids.size(0)

        q_ids = torch.LongTensor([self.sos_id] * batch_size).unsqueeze(1)
        q_ids = q_ids.to(c_ids.device)
        token_type_ids = torch.zeros_like(q_ids)
        position_ids = torch.zeros_like(q_ids)
        q_mask = return_attention_mask(q_ids, self.pad_token_id)
        q_embeddings = self.context_encoder(input_ids=q_ids, attention_mask=q_mask,
                                            token_type_ids=token_type_ids, position_ids=position_ids)[0]

        decoded_q = self.zq_decoder(zq)

        # unroll
        all_q_ids = list()
        all_q_ids.append(q_ids)
        for _ in range(self.max_q_len - 1):
            position_ids = position_ids + 1
            repeated_decoded_q = decoded_q.unsqueeze(1).repeat(1, q_ids.size(1), 1)
            q_outputs = self.question_enc_finetuned(torch.cat((q_embeddings, repeated_decoded_q), dim=-1),
                                                    src_key_padding_mask=q_mask)
            q_outputs = self.question_enc_linear(q_outputs)

            # attention
            mask = c_mask.unsqueeze(1)
            c_attned_by_q, attn_logits = cal_attn(self.question_linear(q_outputs),
                                                  c_outputs, mask)

            # gen logits
            q_concated = torch.cat([q_outputs, c_attned_by_q], dim=2)
            q_concated = self.concat_linear(q_concated)
            q_maxouted, _ = q_concated.view(
                batch_size, 1, self.hidden_size, 2).max(dim=-1)
            gen_logits = self.logit_linear(self.pre_logit_linear(q_maxouted))

            # copy logits
            attn_logits = attn_logits.squeeze(1)
            copy_logits = torch.zeros(
                batch_size, self.ntokens).to(c_ids.device)
            copy_logits = copy_logits - 10000.0
            copy_logits, _ = scatter_max(attn_logits, c_ids, out=copy_logits)
            copy_logits = copy_logits.masked_fill(copy_logits == -10000.0, 0)

            logits = gen_logits + copy_logits.unsqueeze(1)

            q_ids = torch.argmax(logits, 2)
            all_q_ids.append(q_ids)

            q_mask = return_attention_mask(q_ids, self.pad_token_id)
            q_embeddings = self.context_encoder(input_ids=q_ids, attention_mask=q_mask, token_type_ids=token_type_ids,
                                                position_ids=position_ids)[0]

        q_ids = torch.cat(all_q_ids, dim=1)
        q_ids = self.postprocess(q_ids)

        return q_ids

    def top_k_top_p_filtering(self, logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0,
            filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1) -> torch.Tensor:
        """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

    def sample(self, c_ids, a_ids, zq):
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_outputs = self.context_enc_finetuned(c_ids, a_ids)

        batch_size = c_ids.size(0)

        q_ids = torch.LongTensor([self.sos_id] * batch_size).unsqueeze(1)
        q_ids = q_ids.to(c_ids.device)
        token_type_ids = torch.zeros_like(q_ids)
        position_ids = torch.zeros_like(q_ids)
        q_mask = return_attention_mask(q_ids, self.pad_token_id)
        q_embeddings = self.context_encoder(input_ids=q_ids, attention_mask=q_mask,
                                            token_type_ids=token_type_ids, position_ids=position_ids)[0]

        decoded_q = self.zq_decoder(zq)

        # unroll
        all_q_ids = list()
        all_q_ids.append(q_ids)
        for _ in range(self.max_q_len - 1):
            position_ids = position_ids + 1
            repeated_decoded_q = decoded_q.unsqueeze(1).repeat(1, q_ids.size(1), 1)
            q_outputs = self.question_enc_finetuned(torch.cat((q_embeddings, repeated_decoded_q), dim=-1),
                                                    src_key_padding_mask=q_mask)
            q_outputs = self.question_enc_linear(q_outputs)

            # attention
            c_attned_by_q, attn_logits = cal_attn(self.question_linear(q_outputs),
                                                  c_outputs,
                                                  c_mask.unsqueeze(1))

            # gen logits
            q_concated = torch.cat([q_outputs, c_attned_by_q], dim=2)
            q_concated = self.concat_linear(q_concated)
            q_maxouted, _ = q_concated.view(batch_size, 1, self.hidden_size, 2).max(dim=-1)
            gen_logits = self.logit_linear(self.pre_logit_linear(q_maxouted))

            # copy logits
            attn_logits = attn_logits.squeeze(1)
            copy_logits = torch.zeros(batch_size, self.ntokens).to(c_ids.device)
            copy_logits = copy_logits - 10000.0
            copy_logits, _ = scatter_max(attn_logits, c_ids, out=copy_logits)
            copy_logits = copy_logits.masked_fill(copy_logits == -10000.0, 0)

            logits = gen_logits + copy_logits.unsqueeze(1)
            logits = logits.squeeze(1)
            logits = self.top_k_top_p_filtering(logits, 2, 0.8)
            probs = F.softmax(logits, dim=-1)
            q_ids = torch.multinomial(probs, num_samples=1)  # [b,1]
            all_q_ids.append(q_ids)

            q_mask = return_attention_mask(q_ids, self.pad_token_id)
            q_embeddings = self.context_encoder(input_ids=q_ids, attention_mask=q_mask,
                                                token_type_ids=token_type_ids, position_ids=position_ids)[0]

        q_ids = torch.cat(all_q_ids, 1)
        q_ids = self.postprocess(q_ids)

        return q_ids