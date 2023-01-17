import math
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MultiHeadAttention', 'ScaledDotProductAttention']


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1] # query hidden_size
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk) # scores = (N, query_len, kv_len)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value) # (N, query_len, kv_hidden)


class MultiHeadAttention(nn.Module):

    def __init__(self, query_in_features, key_in_features, value_in_features,
                 out_features, num_heads, bias=True, activation=F.gelu):
        """Multi-head attention.
        :param query_in_features: Size of each input sample.
        :param num_heads: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if out_features % num_heads != 0:
            raise ValueError("`out_features`({}) should be divisible by `head_num`({})".format(out_features, num_heads))
        self.out_features = out_features
        self.head_num = num_heads
        self.activation = activation
        self.bias = bias
        self.linear_query = nn.Linear(query_in_features, out_features, bias)
        self.linear_key = nn.Linear(key_in_features, out_features, bias)
        self.linear_value = nn.Linear(value_in_features, out_features, bias)
        self.linear_output = nn.Linear(out_features, out_features, bias)

    def forward(self, queries, keys, values, mask=None):
        """
            mask: size = (N, seq_len) or (N, seq_len, seq_len)
        """
        queries, keys, values = self.linear_query(queries), self.linear_key(keys), self.linear_value(values)
        if self.activation is not None:
            queries = self.activation(queries)
            keys = self.activation(keys)
            values = self.activation(values)

        queries = self._reshape_to_batches(queries)
        keys = self._reshape_to_batches(keys)
        values = self._reshape_to_batches(values)
        if mask is not None:
            if len(mask.size()) == 2:
                mask = torch.matmul(mask.unsqueeze(2), mask.unsqueeze(1))
            mask = mask.repeat(self.head_num, 1, 1)

        y = ScaledDotProductAttention()(queries, keys, values, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_output(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.
        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, out_features = x.size()
        batch_size //= self.head_num
        out_dim = out_features * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, out_features)\
                .permute(0, 2, 1, 3)\
                .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'out_features={}, head_num={}, bias={}, activation={}'.format(
            self.out_features, self.head_num, self.bias, self.activation,
        )