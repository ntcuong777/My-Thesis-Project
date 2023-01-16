import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, bidirectional=False):
        super(CustomLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        if dropout > 0.0 and num_layers == 1:
            dropout = 0.0

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, dropout=dropout,
                            bidirectional=bidirectional, batch_first=True)

    def forward(self, inputs, input_lengths, state=None):
        _, total_length, _ = inputs.size()

        input_packed = pack_padded_sequence(inputs, input_lengths.to("cpu"),
                                            batch_first=True, enforce_sorted=False)

        self.lstm.flatten_parameters()
        output_packed, state = self.lstm(input_packed, state)

        output = pad_packed_sequence(
            output_packed, batch_first=True, total_length=total_length)[0]
        output = self.dropout(output)

        return output, state
