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
        self.infomax_loss = JensenShannonInfoMax(discriminator=nn.Bilinear(hidden_size, hidden_size, 1))

    def forward(self, hidden_states, answer_mask, context_mask, answer_context_weight=0.2):
        answer_mask_mat = torch.matmul(answer_mask.unsqueeze(1), answer_mask.unsqueeze(2))
        answer_context_mask = torch.matmul(answer_mask.unsqueeze(1), context_mask.unsqueeze(2))
        pairwise_loss_info = self.infomax_loss(hidden_states, hidden_states)
        loss_answer_info = (pairwise_loss_info * answer_mask_mat).div(answer_mask_mat.sum())
        loss_answer_context_info = (pairwise_loss_info * answer_context_mask).div(answer_context_mask.sum())
        return (1. - answer_context_weight) * loss_answer_info + answer_context_weight * loss_answer_context_info
