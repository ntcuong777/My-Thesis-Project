import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from math import pi, sqrt, exp


def return_causal_mask_from_sentence_embeds(sentence_embeds):
    """Generate the mask that only uses history data.
    :param sentence_embeds: Input tensor, shape = (N, seq_len, hidden_size).
    :return: The mask.
    """
    batch_size, seq_len, _ = sentence_embeds.size()
    causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(sentence_embeds.device)
    return causal_mask.view(1, seq_len, seq_len).expand(batch_size, -1, -1)


def return_causal_mask_from_position_mask(position_attention_mask):
    """Generate the mask that only uses history data.
    :param position_attention_mask: Input mask indicating which position should be attended to, shape = (N, seq_len).
    :return: The mask.
    """
    pairwise_attention_mask = torch.matmul(position_attention_mask.unsqueeze(2), position_attention_mask.unsqueeze(1))
    causal_mask = torch.tril(pairwise_attention_mask).to(position_attention_mask.device)
    return causal_mask


def freeze_neural_model(network: nn.Module):
    for param in network.parameters():
        param.requires_grad = False


def softargmax(onehot_x, beta=1e4):
    # last dim is the categorical dim, i.e., dim=-1
    categorical_range = torch.arange(onehot_x.size(-1)).to(onehot_x.device).float()
    return torch.sum(F.softmax(onehot_x*beta, dim=-1) * categorical_range, dim=-1).float()


def gaussian_kernel(n=3, sigma=1):
    r = range(-int(n/2), int(n/2)+1)
    return [1 / (sigma * sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]


def sample_gaussian(mu, logvar, num_samples=None):
    if num_samples is None:
        assert len(mu.size()) == 2 and len(
            logvar.size()) == 2  # shape = (batch, dim)
        return mu + torch.randn_like(mu)*torch.exp(0.5 * logvar)
    else:
        assert len(mu.size()) == len(logvar.size())
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if len(mu.size()) == 1:
            return mu.unsqueeze(0) + torch.randn((num_samples, mu.size(0)), device=device)*torch.exp(0.5 * logvar.unsqueeze(0))
        elif len(mu.size()) == 2:
            assert mu.size(0) == 1 and logvar.size(0) == 1
            return mu + torch.randn((num_samples, mu.size(1)), device=device)*torch.exp(0.5 * logvar)


def return_attention_mask(ids: torch.Tensor, pad_token_id=0):
    mask = (ids != pad_token_id).float().to(ids.device)
    return mask


def return_inputs_length(attention_mask: torch.Tensor):
    return attention_mask.sum(dim=-1)


def sample_gumbel(shape, device, eps=1e-10):
    """ Sample from Gumbel(0, 1) """
    U = torch.rand(shape).to(device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor

    # ~Gumbel(0,1), shape=(batch, nza, nzadim)
    gumbels = sample_gumbel(logits.size(), logits.device, eps=eps)
    # ~Gumbel(logits,tau), shape=(batch, nza, nzadim)
    gumbels = (logits + gumbels) / tau
    y_soft = F.softmax(gumbels, dim=dim)  # shape=(batch, nza, nzadim)

    if hard:
        # Straight through.
        _, index = y_soft.max(dim, keepdim=True)  # shape = (batch, nza, 1)
        # sampling one-hot categorical variables
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = (y_hard - y_soft).detach() + y_soft
    else:
        # Re-parametrization trick.
        ret = y_soft
    return ret


def gumbel_latent_var_sampling(num_samples, latent_dim, categorical_dim, device):
    """
    Samples from the latent space and return the corresponding
    image space map.
    :param num_samples: (Int) Number of samples
    :param current_device: (Int) Device to run the model
    :return: (Tensor) with shape (num_samples, latent_dim, categorical_dim)
    """
    # [S x D x Q]
    M = num_samples * latent_dim
    np_y = np.zeros((M, categorical_dim), dtype=np.float32)
    np_y[range(M), np.random.choice(categorical_dim, M)] = 1
    np_y = np.reshape(np_y, [num_samples, latent_dim, categorical_dim])
    z_samples = torch.from_numpy(np_y).to(device)
    return z_samples