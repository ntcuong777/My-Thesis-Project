import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as torch_dist
from torch.distributions.gumbel import Gumbel
from model.model_utils import sample_gaussian, gumbel_softmax, sample_gumbel

# Define MMD loss
def compute_kernel(x, y, latent_dim, kernel_bandwidth, imq_scales=[0.1, 0.2, 0.5, 1.0, 2.0], kernel="rbf"):
    """ Return a kernel of size (batch_x, batch_y) """
    if kernel == "imq":
        Cbase = 2.0 * latent_dim * kernel_bandwidth ** 2
        imq_scales_cuda = torch.tensor(
            imq_scales, dtype=torch.float).cuda()  # shape = (num_scales,)
        # shape = (num_scales, 1, 1)
        Cs = (imq_scales_cuda * Cbase).unsqueeze(1).unsqueeze(2)
        # shape = (batch_x, batch_y)
        k = (Cs / (Cs + torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1).pow(2).unsqueeze(0))).sum(dim=0)
        return k
    elif kernel == "rbf":
        C = 2.0 * latent_dim * kernel_bandwidth ** 2
        return torch.exp(-torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1).pow(2) / C)


def compute_mmd(x, y, latent_dim, kernel_bandwidth=1):
    x_size = x.size(0)
    y_size = y.size(0)
    x_kernel = compute_kernel(x, x, latent_dim, kernel_bandwidth)
    y_kernel = compute_kernel(y, y, latent_dim, kernel_bandwidth)
    xy_kernel = compute_kernel(x, y, latent_dim, kernel_bandwidth)
    mmd_z = (x_kernel - x_kernel.diag().diag()).sum() / ((x_size - 1) * x_size)
    mmd_z_prior = (y_kernel - y_kernel.diag().diag()
                   ).sum() / ((y_size - 1) * y_size)
    mmd_cross = xy_kernel.sum() / (x_size*y_size)
    mmd = mmd_z + mmd_z_prior - 2 * mmd_cross
    return mmd


class VaeGumbelKLLoss(nn.Module):
    def __init__(self):
        super(VaeGumbelKLLoss, self).__init__()

    def forward(self, logits, categorical_dim=10):
        log_ratio = torch.log(logits * categorical_dim + 1e-20)
        KLD = torch.sum(logits * log_ratio, dim=-1).mean()
        return KLD


class GumbelKLLoss(nn.Module):
    def __init__(self):
        super(GumbelKLLoss, self).__init__()

    def forward(self, loc_q, scale_q, loc_p, scale_p):
        g_q = Gumbel(loc_q, scale_q)
        g_p = Gumbel(loc_p, scale_p)
        return torch_dist.kl.kl_divergence(g_q, g_p)


class CategoricalKLLoss(nn.Module):
    def __init__(self):
        super(CategoricalKLLoss, self).__init__()

    def forward(self, P_logits, Q_logits):
        P = F.softmax(P_logits, dim=-1)
        Q = F.softmax(Q_logits, dim=-1)
        log_P = P.log()
        log_Q = Q.log()
        kl = (P * (log_P - log_Q)).sum(dim=-1).sum(dim=-1)
        return kl.mean(dim=0)


class VaeGaussianKLLoss(nn.Module):
    def __init__(self):
        super(VaeGaussianKLLoss, self).__init__()

    def forward(self, mu, logvar):
        sigma = logvar.exp()
        KLD = -0.5 * torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
        return KLD


class GaussianKLLoss(nn.Module):
    def __init__(self):
        super(GaussianKLLoss, self).__init__()

    def forward(self, mu_q, logvar_q, mu_p, logvar_p):
        numerator = logvar_q.exp() + torch.pow(mu_q - mu_p, 2)
        fraction = torch.div(numerator, (logvar_p.exp()))
        kl = 0.5 * torch.sum(logvar_p - logvar_q + fraction - 1, dim=1)
        return kl.mean(dim=0)


class GumbelMMDLoss(nn.Module):
    def __init__(self):
        super(GumbelMMDLoss, self).__init__()

    def forward(self, posterior_z_logits):
        batch_size, latent_dim, nlatent = posterior_z_logits.size()
        prior_z = sample_gumbel((batch_size, latent_dim, nlatent), device=posterior_z.device)
        posterior_z = gumbel_softmax(posterior_z_logits, hard=False)
        return compute_mmd(posterior_z, prior_z, latent_dim)


class ContinuousKernelMMDLoss(nn.Module):
    def __init__(self):
        super(ContinuousKernelMMDLoss, self).__init__()

    def forward(self, posterior_z):
        # input shape = (batch, dim)
        batch_size, latent_dim = posterior_z.size()
        prior_z = torch.randn(batch_size, latent_dim).to(posterior_z.device)
        return compute_mmd(posterior_z, prior_z, latent_dim)


class GaussianJensenShannonDivLoss(nn.Module):
    def __init__(self):
        super(GaussianJensenShannonDivLoss, self).__init__()
        self.gaussian_kl_loss = GaussianKLLoss()

    def forward(self, mu1, logvar1, mu2, logvar2):
        mean_mu, mean_logvar = (mu1+mu2) / 2, ((logvar1.exp() + logvar2.exp()) / 2).log()

        loss = self.gaussian_kl_loss(mu1, logvar1, mean_mu, mean_logvar)
        loss += self.gaussian_kl_loss(mu2, logvar2, mean_mu, mean_logvar)
     
        return (0.5 * loss)


class CategoricalJensenShannonDivLoss(nn.Module):
    def __init__(self):
        super(CategoricalJensenShannonDivLoss, self).__init__()

    def forward(self, posterior_za_logits, prior_za_logits):
        posterior_za_probs = F.softmax(posterior_za_logits, dim=1)
        prior_za_probs = F.softmax(prior_za_logits, dim=1)

        mean_probs = 0.5 * (posterior_za_probs + prior_za_probs)
        loss = F.kl_div(F.log_softmax(posterior_za_logits,
                         dim=1), mean_probs, reduction="batchmean")
        loss += F.kl_div(F.log_softmax(prior_za_logits, dim=1),
                         mean_probs, reduction="batchmean")

        return (0.5 * loss)
