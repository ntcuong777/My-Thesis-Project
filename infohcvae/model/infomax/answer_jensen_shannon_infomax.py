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

    def forward(self, hidden_states, answer_mask, context_mask, answer_context_weight=0.25):
        answer_mask = answer_mask.type(torch.FloatTensor).to(hidden_states.device)
        context_mask = context_mask.type(torch.FloatTensor).to(hidden_states.device)

        answer_embs = hidden_states * answer_mask
        mean_answer_emb = answer_embs.div(answer_mask.sum(dim=-1, keepdims=True))
        answer_span_loss_info = self.answer_span_loss(mean_answer_emb.unsqueeze(1), answer_embs)

        mean_context_emb = (hidden_states * context_mask).div(context_mask.sum(dim=-1, keepdims=True))
        answer_context_loss_info = self.answer_context_loss(mean_answer_emb, mean_context_emb)

        return (1. - answer_context_weight) * answer_span_loss_info + answer_context_weight * answer_context_loss_info
