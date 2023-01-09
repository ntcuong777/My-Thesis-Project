import torch.nn as nn
from transformers import T5EncoderModel, T5Config

class T5ContextAnswerEncoder(nn.Module):
    def __init__(self, base_model="t5-base", num_enc_finetune_layers=2):
        super(T5ContextAnswerEncoder, self).__init__()

        config = T5Config.from_pretrained(base_model)
        self.encoder = T5EncoderModel.from_pretrained(base_model)

        # Freeze all layers
        for param in self.t5_encoder.parameters():
            param.requires_grad = False
        # Only some top layers are for fine-tuning
        for idx in range(config.num_layers - num_enc_finetune_layers, config.num_layers):
            for param in self.t5_encoder.get_encoder().block[idx].parameters():
                param.requires_grad = True

    def forward(self, context_ids, context_mask, answer_mask):
        # context enc
        c_hidden_states = self.encoder(input_ids=context_ids, attention_mask=context_mask)[0]

        # context and answer enc
        a_with_cls_mask = answer_mask.detach()
        a_with_cls_mask[:, 0] = 1  # attentive to [CLS] token
        a_ids = context_ids * a_with_cls_mask
        a_hidden_states = self.t5_encoder(input_ids=a_ids, attention_mask=a_with_cls_mask)[0]
        c_a_hidden_states = c_hidden_states + a_hidden_states
        return c_hidden_states, c_a_hidden_states
