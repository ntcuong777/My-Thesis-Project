import torch
import torch.nn as nn
from .jensen_shannon_infomax import JensenShannonInfoMax


class AnswerJensenShannonInfoMax(nn.Module):
    """discriminator network.
    Args:
        x_dim (int): input dim, for example m x n x c for [m, n, c]
        y_dim (int): dimension of latent code (typically a number in [10 - 256])
    """

    def __init__(self, hidden_size=768):
        super(AnswerJensenShannonInfoMax, self).__init__()
        self.answer_span_loss = JensenShannonInfoMax(discriminator=nn.Bilinear(hidden_size, hidden_size, 1))
        self.answer_context_loss = JensenShannonInfoMax(discriminator=nn.Bilinear(hidden_size, hidden_size, 1))

    def forward(self, hidden_states, answer_mask, start_mask, end_mask, context_mask, answer_context_weight=0.25):
        answer_mask = answer_mask.type(torch.FloatTensor).to(hidden_states.device)
        context_mask = context_mask.type(torch.FloatTensor).to(hidden_states.device)

        answer_embs = hidden_states * answer_mask.unsqueeze(2)
        mean_answer_emb = answer_embs.sum(dim=1).div(answer_mask.sum(dim=-1, keepdims=True))
        start_emb = (hidden_states * start_mask.unsqueeze(2)).sum(dim=1)
        end_emb = (hidden_states * end_mask.unsqueeze(2)).sum(dim=1)
        answer_span_loss_info = 0.5 * (self.answer_span_loss(start_emb, mean_answer_emb) +
                                       self.answer_span_loss(end_emb, mean_answer_emb))

        context_embs = hidden_states * context_mask.unsqueeze(2)
        mean_context_emb = context_embs.sum(dim=1).div(context_mask.sum(dim=-1, keepdims=True))
        answer_context_loss_info = (0.5 * self.answer_context_loss(mean_answer_emb, mean_context_emb) +
                                    0.25 * self.answer_context_loss(start_emb, mean_context_emb) +
                                    0.25 * self.answer_context_loss(end_emb, mean_context_emb))

        return (1. - answer_context_weight) * answer_span_loss_info + answer_context_weight * answer_context_loss_info
