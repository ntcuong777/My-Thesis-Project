import torch.nn as nn
from transformers import T5EncoderModel, T5Config


class CustomT5Encoder(nn.Module):
    def __init__(self, base_model="t5-base", num_enc_finetune_layers=2):
        super(CustomT5Encoder, self).__init__()

        config = T5Config.from_pretrained(base_model)
        self.t5_encoder = T5EncoderModel.from_pretrained(base_model)

        # Freeze all layers
        for param in self.t5_encoder.parameters():
            param.requires_grad = False
        # Only some top layers are for fine-tuning
        for idx in range(config.num_layers - num_enc_finetune_layers, config.num_layers):
            for param in self.t5_encoder.get_encoder().block[idx].parameters():
                param.requires_grad = True

    def forward(self, input_ids, input_mask, answer_mask=None, return_input_hidden_states=None):
        """
            Encode the tokens in `input_ids` given mask `input_mask`,
            If the encoder is used for aggregate the `input_ids` with
            a specific text span representing an answer to a question,
            then provide the `answer_mask` explicitly.
        """
        # input enc
        input_hidden_states = self.t5_encoder(input_ids=input_ids, attention_mask=input_mask)[0]

        if answer_mask is not None:
            # input and answer enc
            a_with_cls_mask = answer_mask.detach()
            a_with_cls_mask[:, 0] = 1  # attentive to [CLS] token
            a_ids = input_ids * a_with_cls_mask
            a_hidden_states = self.t5_encoder(input_ids=a_ids, attention_mask=a_with_cls_mask)[0]
            input_with_a_hidden_states = input_hidden_states + a_hidden_states
            if return_input_hidden_states is not None and return_input_hidden_states:
                return input_hidden_states, input_with_a_hidden_states
            else:
                return input_with_a_hidden_states

        return input_hidden_states
