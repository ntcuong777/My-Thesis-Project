import warnings
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import Seq2SeqModelOutput
from transformers import T5Config, T5Model

# Warning message for FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


class T5ModelAnswerDecoder(T5Model):
    def __init__(self, config: T5Config, nzadim, nza_values):
        super().__init__(config)

        self.embed_size_per_head = config.d_model // config.num_heads
        self.memory_projection = nn.Linear(
            nzadim * nza_values,
            config.num_decoder_layers * config.num_heads * self.embed_size_per_head,
            bias=False,
        )

    def __post_init__(self):
        """ We not gonna use the encoder """
        self.__remove_encoder()

    def __remove_encoder(self):
        self.encoder = None

    def build_past(self, za):
        projection = self.memory_projection(za, dim=-1)
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
            encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            decoder_inputs_embeds: Optional[torch.Tensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqModelOutput]:
        r""" Here, `input_ids` is the `c_ids` in AnswerDecoder

        Returns:

        Example:

        ```python
        >>> from transformers import T5Tokenizer, T5Model

        >>> tokenizer = T5Tokenizer.from_pretrained("t5-small")
        >>> model = T5Model.from_pretrained("t5-small")

        >>> input_ids = tokenizer(
        ...     "Studies have been shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

        >>> # preprocess: Prepend decoder_input_ids with start token which is pad token for T5Model.
        >>> # This is not needed for torch's T5ForConditionalGeneration as it does this internally using labels arg.
        >>> decoder_input_ids = model._shift_right(decoder_input_ids)

        >>> # forward pass
        >>> outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
        >>> last_hidden_states = outputs.last_hidden_state
        ```"""
        assert "sampled_za" in kwargs, "For answer decoding, `sampled_za` must not be None"
        assert "context_ids" in kwargs, "For answer decoding, `context_ids` must not be None"
        assert "context_mask" in kwargs, "For answer decoding, `context_mask` must not be None"

        sampled_za = kwargs["sampled_za"]
        context_ids = kwargs["context_ids"]
        context_mask = kwargs["context_mask"]

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        """ CUSTOM START: remove encoding part """
        hidden_states = None
        """ CUSTOM END: remove encoding part """

        # CUSTOM: initialize `past_key_values` with `za` for question generation
        if past_key_values is None:
            past_key_values = self.build_past(sampled_za)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            """ CUSTOM START: check if `hidden_states` is None before changing `device` """
            if hidden_states is not None:
                hidden_states = hidden_states.to(self.decoder.first_device)
            """ CUSTOM END: check if `hidden_states` is None """
            if context_ids is not None: # replace `decoder_input_ids` with `context_ids`
                context_ids = context_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if context_mask is not None: # replace `decoder_attention_mask` with `context_mask`
                context_mask = context_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=context_ids, # replace `decoder_input_ids` with `context_ids`
            attention_mask=context_mask, # replace `decoder_attention_mask` with `context_mask`
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

        """ CUSTOM START: output of model does not contain data from `encoder_outputs` """
        if not return_dict:
            return decoder_outputs # + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=None,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )
        """ CUSTOM END: output of model does not contain data from `encoder_outputs` """
