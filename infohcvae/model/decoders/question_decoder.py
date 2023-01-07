import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from infohcvae.model.model_utils import return_attention_mask, cal_attn
from transformers import T5Config, T5EncoderModel
from infohcvae.model.infomax.dim_bce_infomax import DimBceInfoMax
from infohcvae.model.custom import T5ForConditionalGenerationDecoderOnly


class _ContextEncoderforQG(nn.Module):
    def __init__(self, config: T5Config, pad_id, base_model="t5-base", n_finetune_layers=2):
        super(_ContextEncoderforQG, self).__init__()
        self.t5_encoder = T5EncoderModel.from_pretrained(base_model)
        self.pad_token_id = pad_id

        # Freeze all layers
        for param in self.t5_encoder.parameters():
            param.requires_grad = False
        # Only some top layers are for fine-tuning
        for idx in range(config.num_layers - n_finetune_layers, config.num_layers):
            for param in self.t5_encoder.get_encoder().block[idx].parameters():
                param.requires_grad = True

        self.context_linear = nn.Linear(2 * config.d_model, 2 * config.d_model)
        self.fusion = nn.Linear(4 * config.d_model, config.d_model, bias=False)
        self.gate = nn.Linear(4 * config.d_model, config.d_model, bias=False)

    def forward(self, c_ids, a_mask):
        c_mask = return_attention_mask(c_ids, self.pad_token_id)

        # context enc
        c_hidden_states = self.t5_encoder(input_ids=c_ids, attention_mask=c_mask)[0]

        # context and answer enc
        a_ids = c_ids * a_mask
        a_hidden_states = self.t5_encoder(input_ids=a_ids, attention_mask=a_mask)[0]
        a_attned_by_c, _ = cal_attn(a_hidden_states, c_hidden_states,
                                    c_mask.unsqueeze(1))  # output shape = (N, seq_len, d_model)
        c_attned_by_a, _ = cal_attn(c_hidden_states, a_hidden_states,
                                    a_mask.unsqueeze(1))  # output shape = (N, seq_len, d_model)
        c_a_hidden_states = torch.cat([a_attned_by_c, c_attned_by_a], dim=-1)

        # gated attention
        mask = torch.matmul(c_mask.unsqueeze(2), c_mask.unsqueeze(1))
        c_attned_by_c, _ = cal_attn(self.context_linear(c_a_hidden_states),
                                    c_a_hidden_states, mask)
        c_concat = torch.cat([c_a_hidden_states, c_attned_by_c], dim=2)
        c_fused = self.fusion(c_concat).tanh()
        c_gate = self.gate(c_concat).sigmoid()
        c_outputs = c_gate * c_fused + (1 - c_gate) * c_a_hidden_states
        return c_outputs


class QuestionDecoder(nn.Module):
    def __init__(self, sos_id, eos_id, pad_id, nzqdim,
                 num_dec_finetune_layers=2, base_model="t5-base", max_q_len=64):
        super(QuestionDecoder, self).__init__()

        config = T5Config.from_pretrained(base_model)

        self.t5_question_decoder = T5ForConditionalGenerationDecoderOnly.from_pretrained(base_model, nzqdim=nzqdim)
        # Freeze all decoder layers
        for param in self.t5_question_decoder.parameters():
            param.requires_grad = False
        # Only some top layers are for fine-tuning
        for idx in range(config.num_layers - num_dec_finetune_layers, config.num_layers):
            for param in self.t5_question_decoder.get_decoder().block[idx].parameters():
                param.requires_grad = True

        self.sos_id = sos_id
        self.eos_id = eos_id
        self.num_dec_finetune_layers = num_dec_finetune_layers
        # this max_len include sos eos
        self.max_q_len = max_q_len
        self.pad_token_id = pad_id

        self.context_enc_finetuned = _ContextEncoderforQG(config, pad_id, base_model=base_model,
                                                          n_finetune_layers=num_dec_finetune_layers)

        self.infomax_est = DimBceInfoMax(config.d_model, config.d_model, use_billinear=True)

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

    def forward(self, c_ids, q_ids, a_mask, zq):
        c_hidden_states = self.context_enc_finetuned(c_ids, a_mask)

        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        q_mask = return_attention_mask(q_ids, self.pad_token_id)

        # question dec
        q_outputs = self.t5_question_decoder(sampled_zq=zq, input_ids=q_ids,
                                             attention_mask=q_mask, context_ids=c_ids,
                                             context_embeds=c_hidden_states, context_mask=c_mask)
        logits = q_outputs[1]
        q_maxouted = q_outputs[3]

        if self.training:
            # mutual information btw answer and question (customized: use bi-lstm to average the question & answer)
            a_emb = c_hidden_states * a_mask.float().unsqueeze(2)
            a_mean_emb = torch.sum(a_emb, dim=1) / a_mask.sum(1).unsqueeze(1).float()

            q_emb = q_maxouted * q_mask.unsqueeze(2)
            q_mean_emb = torch.sum(q_emb, dim=1) / q_mask.sum(dim=1, keepdim=True).float()

            return logits, self.infomax_est(q_mean_emb, a_mean_emb)
        else:
            return logits

    def generate(self, c_ids, a_mask, zq):
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_hidden_states = self.context_enc_finetuned(c_ids, a_mask)

        batch_size = c_ids.size(0)

        q_ids = torch.LongTensor([self.sos_id] * batch_size).unsqueeze(1)
        q_ids = q_ids.to(c_ids.device)
        q_mask = return_attention_mask(q_ids, self.pad_token_id)

        # unroll
        all_q_ids = list()
        all_q_ids.append(q_ids)
        for _ in range(self.max_q_len - 1):
            logits = self.t5_question_decoder(sampled_zq=zq, input_ids=q_ids,
                                              attention_mask=q_mask, context_ids=c_ids,
                                              context_embeds=c_hidden_states, context_mask=c_mask)[1]
            q_ids = torch.argmax(logits, dim=2)
            all_q_ids.append(q_ids)
            q_mask = return_attention_mask(q_ids, self.pad_token_id)

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

    def sample(self, c_ids, a_mask, zq):
        c_mask = return_attention_mask(c_ids, self.pad_token_id)
        c_hidden_states = self.context_enc_finetuned(c_ids, a_mask)

        batch_size = c_ids.size(0)

        q_ids = torch.LongTensor([self.sos_id] * batch_size).unsqueeze(1)
        q_ids = q_ids.to(c_ids.device)
        q_mask = return_attention_mask(q_ids, self.pad_token_id)

        # unroll
        all_q_ids = list()
        all_q_ids.append(q_ids)
        for _ in range(self.max_q_len - 1):
            logits = self.t5_question_decoder(sampled_zq=zq, input_ids=q_ids,
                                              attention_mask=q_mask, context_ids=c_ids,
                                              context_embeds=c_hidden_states, context_mask=c_mask)[1]
            logits = logits.squeeze(1)
            logits = self.top_k_top_p_filtering(logits, 2, 0.8)
            probs = F.softmax(logits, dim=-1)
            q_ids = torch.multinomial(probs, num_samples=1)  # [b,1]
            all_q_ids.append(q_ids)
            q_mask = return_attention_mask(q_ids, self.pad_token_id)

        q_ids = torch.cat(all_q_ids, 1)
        q_ids = self.postprocess(q_ids)
        return q_ids
