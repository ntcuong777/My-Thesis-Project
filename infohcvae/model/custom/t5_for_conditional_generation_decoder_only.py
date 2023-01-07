import warnings

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch_scatter import scatter_max
from transformers import T5ForConditionalGeneration, T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput
from infohcvae.model.model_utils import cal_attn

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


class T5ForConditionalGenerationDecoderOnly(T5ForConditionalGeneration):
    def __init__(self, config: T5Config, nzqdim, dropout=0.3):
        super().__init__(config)
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

    def __post_init__(self):
        """ We not gonna use the encoder """
        self.__remove_encoder()

    def __remove_encoder(self):
        self.encoder = None

    def forward(
        self,
        sampled_zq=None,
        input_ids=None,
        attention_mask=None,
        context_ids=None,
        context_embeds=None,
        context_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """ Here, `input_ids` is the `q_ids`, `attention_mask` is the `q_mask` in QuestionDecoder """

        assert sampled_zq is not None, "For question decoding, `sampled_zq` must not be None"
        assert context_ids is not None, "For question decoding, `context_ids` must not be None"
        assert context_embeds is not None, "For question decoding, `context_embeds` must not be None"
        assert context_mask is not None, "For question decoding, `context_mask` must not be None"
        assert attention_mask is not None, "For question decoding, `attention_mask` must not be None"

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        """ CUSTOM START: Remove encoding part """
        hidden_states = None
        """ CUSTOM END: Remove encoding part """

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        """ CUSTOM START: initialize `past_key_values` with `zq` for question generation """
        if past_key_values is None:
            past_key_values = self.build_past(sampled_zq)
        """ CUSTOM END: initialize `past_key_values` with `zq` for question generation """

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None and labels is None:
            # assert (
            #    labels is None
            # ), "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            """ CUSTOM START: check if `hidden_states` is None before changing `device` """
            if hidden_states is not None:
                hidden_states = hidden_states.to(self.decoder.first_device)
            """ CUSTOM END: check if `hidden_states` is None """
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
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
        batch_size, max_q_len = input_ids.size()

        # attention
        # For attention calculation, linear layer is there for projection
        mask = torch.matmul(attention_mask.unsqueeze(2), context_mask.unsqueeze(1))
        c_attned_by_q, attn_logits = cal_attn(self.question_linear(sequence_output),
                                              context_embeds, mask)

        # gen logits
        q_concated = torch.cat([sequence_output, c_attned_by_q], dim=2)
        q_concated = self.concat_linear(q_concated)
        q_maxouted, _ = q_concated.view(
            batch_size, max_q_len, self.d_model, 2).max(dim=-1)
        gen_logits = self.lm_head(q_maxouted)

        # copy logits
        bq = batch_size * max_q_len
        context_ids = context_ids.unsqueeze(1).repeat(
            1, max_q_len, 1).view(bq, -1).contiguous()
        attn_logits = attn_logits.view(bq, -1).contiguous()
        copy_logits = torch.zeros(bq, self.ntokens).to(context_ids.device)
        copy_logits = copy_logits - 10000.0
        copy_logits, _ = scatter_max(attn_logits, context_ids, out=copy_logits)
        copy_logits = copy_logits.masked_fill(copy_logits == -10000.0, 0)
        copy_logits = copy_logits.view(batch_size, max_q_len, -1).contiguous()

        if self.training:
            lm_logits = gen_logits + copy_logits
        else: # using model for generation
            lm_logits = gen_logits + copy_logits.unsqueeze(1)
        """ CUSTOM END: Question decoding language modeling head specific to this QAG problem """

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        """
        CUSTOM START:
        - Output of model does not contain data from `encoder_outputs`
        - Return `q_maxouted` as `decoder_hidden_states`
        """
        if not return_dict:
            output = (lm_logits,) + (None, q_maxouted, None, None) # decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        out = Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=None,
            decoder_hidden_states=q_maxouted,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )
        return out
        """ CUSTOM END: output of model does not contain data from `encoder_outputs` """

    def build_past(self, z):
        projection = self.memory_projection(z)
        cross_attn = projection.reshape(
            self.config.num_decoder_layers,
            projection.shape[0],
            self.config.num_heads,
            1,
            self.embed_size_per_head,
        )
        past_key_values = tuple((ca, ca) for ca in cross_attn)
        return past_key_values
