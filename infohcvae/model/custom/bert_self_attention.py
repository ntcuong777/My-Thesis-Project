import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, activation=F.gelu, dropout=0.0):
        super().__init__()
        assert hidden_size % num_attention_heads == 0,\
            "The hidden size is not a multiple of the number of attention heads"

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dense = nn.Linear(hidden_size, hidden_size)

        self.activation = activation
        self.dropout = None
        if dropout > 0.0:
            self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_key_layer = self.key(hidden_states)  # [Batch_size x Seq_length x Hidden_size]
        mixed_value_layer = self.value(hidden_states)  # [Batch_size x Seq_length x Hidden_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [N x num_heads x seq_len x head_size]
        key_layer = self.transpose_for_scores(mixed_key_layer)  # [N x num_heads x seq_len x head_size]
        value_layer = self.transpose_for_scores(mixed_value_layer) # [N x num_heads x seq_len x head_size]

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [N x num_heads x Seq_len x seq_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) # [N x num_heads x seq_len x seq_len]

        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.size(0), hidden_states.size(1))

        extended_attention_mask = None
        if len(attention_mask.size()) == 2:
            extended_attention_mask = torch.matmul(attention_mask.unsqueeze(2), attention_mask.unsqueeze(1))\
                .unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)
        else:
            extended_attention_mask = attention_mask.unsqueeze(1).expand(-1, self.num_attention_heads, -1, -1)

        # attend to only some positions
        attention_scores = extended_attention_mask * attention_scores - (1 - extended_attention_mask) * 1e9

        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [N x Num_of_heads x Seq_length x Seq_length]
        context_layer = torch.matmul(attention_probs, value_layer)  # [N x num_heads x seq_len x head_size]

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # [N x seq_len x num_heads x head_size]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,) # [N x seq_len x hidden_size]
        context_layer = context_layer.view(*new_context_layer_shape)  # [Batch_size x Seq_length x Hidden_size]

        output = self.dense(context_layer)
        if self.activation is not None:
            output = self.activation(output)

        if self.dropout is not None:
            output = self.dropout(output)

        return output
