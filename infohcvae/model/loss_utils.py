import torch


def imq_kernel(z1, z2, kernel_bandwidth=2.0, scales=[0.1, 0.2, 0.5, 1.0, 2.0]):
    """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""

    Cbase = 2.0 * z1.size(-1) * kernel_bandwidth ** 2
    k = 0
    for scale in scales:
        C = scale * Cbase
        k += C / (C + torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2)
    return k


def rbf_kernel(z1, z2, kernel_bandwidth=2.0):
    """Returns a matrix of shape [batch x batch] containing the pairwise kernel computation"""

    C = 2.0 * z1.size(-1) * kernel_bandwidth ** 2
    k = torch.exp(-torch.norm(z1.unsqueeze(1) - z2.unsqueeze(0), dim=-1) ** 2 / C)
    return k


def compute_mmd(posterior_z, prior_z, kernel_type="imq"):
    N = posterior_z.shape[0]  # batch size

    if kernel_type == "rbf":
        k_z = rbf_kernel(posterior_z, posterior_z)
        k_z_prior = rbf_kernel(prior_z, prior_z)
        k_cross = rbf_kernel(posterior_z, prior_z)
    else:
        k_z = imq_kernel(posterior_z, posterior_z)
        k_z_prior = imq_kernel(prior_z, prior_z)
        k_cross = imq_kernel(posterior_z, prior_z)

    mmd_z = (k_z - k_z.diag().diag()).sum() / ((N - 1) * N)
    mmd_z_prior = (k_z_prior - k_z_prior.diag().diag()).sum() / ((N - 1) * N)
    mmd_cross = k_cross.sum() / (N ** 2)

    mmd_loss = mmd_z + mmd_z_prior - 2 * mmd_cross

    return mmd_loss
