import torch
import torch.nn as nn
import torch.nn.functional as F


class _AnswerSpanInfoMaxDiscriminator(nn.Module):
    def __init__(self, feature_size):
        super(_AnswerSpanInfoMaxDiscriminator, self).__init__()
        self.bilinear = nn.Bilinear(feature_size, feature_size, 1)
        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, global_enc, local_enc):
        global_enc = global_enc.unsqueeze(1) # (batch, 1, global_size)
        global_enc = global_enc.expand(-1, local_enc.size(1), -1) # (batch, seq_len, global_size)
        # (batch, seq_len, global_size) * (batch, seq_len, local_size) -> (batch, seq_len, 1)
        scores = self.bilinear(global_enc.contiguous(), local_enc.contiguous())

        return scores


class AnswerSpanInfoMaxLoss(nn.Module):
    '''
    Deep infomax loss for SQuAD question answering dataset.
    As the difference between GC and LC only lies in whether we do summarization over x,
    this class can be used as both GC and LC.
    '''
    def __init__(self, feature_size):
        super(AnswerSpanInfoMaxLoss, self).__init__()
        self.discriminator = _AnswerSpanInfoMaxDiscriminator(feature_size)
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.dropout = nn.Dropout(0.5)

    def summarize(self, x):
        return x.mean(dim=1).sigmoid()

    def forward(self, x_enc, x_fake, y_enc, y_fake, do_summarize=False):
        '''
        Args:
            global_enc, local_enc: (batch, seq, dim)
            global_fake, local_fake: (batch, seq, dim)
        '''
        # Compute g(x, y)
        if do_summarize:
            x_enc = self.summarize(x_enc)
        x_enc = self.dropout(x_enc)
        y_enc = self.dropout(y_enc)
        logits = self.discriminator(x_enc, y_enc)
        batch_size1, n_seq1 = y_enc.size(0), y_enc.size(1)
        labels = torch.ones(batch_size1, n_seq1)

        # Compute 1 - g(x, y^(\bar))
        y_fake = self.dropout(y_fake)
        _logits = self.discriminator(x_enc, y_fake)
        batch_size2, n_seq2 = y_fake.size(0), y_fake.size(1)
        _labels = torch.zeros(batch_size2, n_seq2)

        logits, labels = torch.cat((logits, _logits), dim=1), torch.cat((labels, _labels), dim=1)

        # Compute 1 - g(x^(\bar), y)
        if do_summarize:
            x_fake = self.summarize(x_fake)
        x_fake = self.dropout(x_fake)
        _logits = self.discriminator(x_fake, y_enc)
        _labels = torch.zeros(batch_size1, n_seq1)

        logits, labels = torch.cat((logits, _logits), dim=1), torch.cat((labels, _labels), dim=1)

        loss = self.bce_loss(logits.squeeze(2), labels.cuda())

        return loss