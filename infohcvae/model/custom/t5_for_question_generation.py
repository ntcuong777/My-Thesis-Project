from typing import Optional, Tuple, Union
import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch_scatter import scatter_max
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import (
    Seq2SeqLMOutput,
    BaseModelOutput,
)
from infohcvae.model.model_utils import cal_attn
from .custom_t5_encoder import CustomT5Encoder

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


class CustomT5ForQuestionGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, custom_encoder: CustomT5Encoder, nzqdim, n_finetune_layers, dropout=0.3):
        super().__init__(config)

        self.config = config
        self.nzqdim = nzqdim

        # Freeze all layers
        for param in self.parameters():
            param.requires_grad = False
        # Only some top layers of encoder & decoders are for fine-tuning
        for idx in range(config.num_layers - n_finetune_layers, config.num_layers):
            for param in self.decoder.block[idx].parameters():
                param.requires_grad = True

        # Freeze language modeling head
        for param in self.lm_head.parameters():
            param.requires_grad = False

        # Change the T5Stack definition of `encoder` to the project's customed implementation
        # for encoding context and answer
        self.encoder = custom_encoder

        self.d_model = config.d_model
        self.ntokens = config.vocab_size
        self.embed_size_per_head = config.d_model // config.num_heads
        self.memory_projection = nn.Linear(
            nzqdim,
            config.num_decoder_layers * config.num_heads * self.embed_size_per_head,
            bias=False,
        )

        self.question_linear = nn.Linear(config.d_model, config.d_model)
        self.concat_linear = nn.Sequential(nn.Linear(2 * config.max_length, 2 * config.d_model),
                                           nn.Dropout(dropout),
                                           nn.Linear(2 * config.d_model, 2 * config.d_model))

    def build_past(self, zq):
        projection = self.memory_projection(zq)
        cross_attn = projection.reshape(
            self.config.num_decoder_layers,
            projection.shape[0],
            self.config.num_heads,
            1,
            self.embed_size_per_head,
        )
        past_key_values = tuple((ca, ca) for ca in cross_attn)
        return past_key_values

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        assert "sampled_zq" in kwargs, "For question decoding, `sampled_zq` must not be None"
        assert "context_ids" in kwargs, "For question decoding, `context_ids` must not be None"
        assert "context_mask" in kwargs, "For question decoding, `context_mask` must not be None"
        assert "answer_mask" in kwargs, "For question decoding, `context_mask` must not be None"
        assert "question_ids" in kwargs, "For question decoding, `question_ids` must not be None"
        assert "question_mask" in kwargs, "For question decoding, `question_mask` must not be None"

        sampled_zq = kwargs["sampled_zq"]
        context_ids = kwargs["context_ids"]
        context_mask = kwargs["context_mask"]
        answer_mask = kwargs["answer_mask"]
        question_ids = kwargs["question_ids"]
        question_mask = kwargs["question_mask"]

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # CUSTOM: Encode context & answer to have `encoder_outputs`
        # Convert encoder inputs in embeddings if needed
        c_a_hidden_states = self.encoder(context_ids=context_ids, context_mask=context_mask, answer_mask=answer_mask)

        if not return_dict:
            encoder_outputs = BaseModelOutput(last_hidden_state=c_a_hidden_states,
                                              hidden_states=None,
                                              attentions=None)
        else:
            encoder_outputs = (c_a_hidden_states, None, None)

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        # CUSTOM: `decoder_input_ids` changed to `question_ids`, `decoder_input_embeds` is removed
        if (
                labels is not None
                and question_ids is None
        ):
            # get decoder inputs from shifting lm labels to the right
            question_ids = self._shift_right(labels)

        # CUSTOM: initialize `past_key_values` with `zq` for question generation
        if past_key_values is None:
            past_key_values = self.build_past(sampled_zq)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            # CUSTOM START: `decoder_input_ids` is changed to `question_ids`
            if question_ids is not None:
                question_ids = question_ids.to(self.decoder.first_device)
            # CUSTOM: `attention_mask` is modified into `context_mask` """
            if context_mask is not None:
                context_mask = context_mask.to(self.decoder.first_device)
            # CUSTOM START: `decoder_attention_mask` is mofified into `question_mask`
            if question_mask is not None:
                question_mask = question_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=question_ids, # CUSTOM: decode `question_ids`
            attention_mask=question_mask, # CUSTOM: use `question_mask`
            inputs_embeds=None,  # CUSTOM: set to none, since we dont need to use this
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=context_mask, # CUSTOM: set to `context_mask`
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=None,  # CUSTOM: set to none, we dont need to use this
            output_hidden_states=None,  # CUSTOM: set to none, we dont need to use this
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        """ CUSTOM START: Question decoding language modeling head specific to this QAG problem """
        batch_size, max_q_len = question_ids.size()
        # gen logits
        gen_logits = self.lm_head(sequence_output)

        # copy logits
        # context-question attention
        mask = torch.matmul(question_mask.unsqueeze(2), context_mask.unsqueeze(1))
        _, attn_logits = cal_attn(sequence_output, c_a_hidden_states, mask)

        bq = batch_size * max_q_len
        context_ids = context_ids.unsqueeze(1).repeat(
            1, max_q_len, 1).view(bq, -1).contiguous()
        attn_logits = attn_logits.view(bq, -1).contiguous()
        copy_logits = torch.zeros(bq, self.ntokens).to(context_ids.device) - 10000.0
        copy_logits, _ = scatter_max(attn_logits, context_ids, out=copy_logits)
        copy_logits = copy_logits.masked_fill(copy_logits == -10000.0, 0)
        copy_logits = copy_logits.view(batch_size, max_q_len, -1).contiguous()

        lm_logits = gen_logits + copy_logits
        """ CUSTOM END: Question decoding language modeling head specific to this QAG problem """

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        """
        CUSTOM START:
        - Return `sequence_output` (last hidden states) as `decoder_hidden_states` for computing InfoMax objective
        """
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        out = Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=encoder_outputs[0],
            encoder_hidden_states=None,
            encoder_attentions=None,
        )
        return out

