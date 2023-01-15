import torch


def compute_kernel(x1, x2, kernel_type="imq", exclude_diag: bool = None):
    if kernel_type == "rbf":
        result = compute_rbf(x1, x2)
    elif kernel_type == "imq":
        assert exclude_diag is not None, "IMQ kernel requires `exclude_diag` argument"
        result = compute_inv_mult_quad(x1, x2, exclude_diag=exclude_diag)
    else:
        raise ValueError('Undefined kernel type.')

    return result


def compute_rbf(x1, x2):
    """
    Computes the RBF Kernel between x1 and x2.
    :param x1: (Tensor)
    :param x2: (Tensor)
    :param eps: (Float)
    :return:
    """
    dim1_1, dim1_2 = x1.size(0), x2.size(0)
    depth = x1.size(1)
    x1 = x1.view(dim1_1, 1, depth)
    x2 = x2.view(1, dim1_2, depth)
    x1_core = x1.expand(dim1_1, dim1_2, depth)
    x2_core = x2.expand(dim1_1, dim1_2, depth)
    numerator = (x1_core - x2_core).pow(2).mean(2) / depth
    return torch.exp(-numerator)


def compute_inv_mult_quad(z1, z2, z_var=2., exclude_diag=True):
    r"""Calculate sum of sample-wise measures of inverse multiquadratics kernel described in the WAE paper.
    Args:
        z1 (Tensor): batch of samples from a multivariate gaussian distribution \
            with scalar variance of z_var.
        z2 (Tensor): batch of samples from another multivariate gaussian distribution \
            with scalar variance of z_var.
        exclude_diag (bool): whether to exclude diagonal kernel measures before sum it all.
    """
    assert z1.size() == z2.size()
    assert z1.ndimension() == 2

    z_dim = z1.size(1)
    C = 2*z_dim*z_var

    z11 = z1.unsqueeze(1).repeat(1, z2.size(0), 1)
    z22 = z2.unsqueeze(0).repeat(z1.size(0), 1, 1)

    kernel_matrix = C/(1e-9+C+(z11-z22).pow(2).sum(2))
    kernel_sum = kernel_matrix.sum()
    # numerically identical to the formulation. but..
    if exclude_diag:
        kernel_sum -= kernel_matrix.diag().sum()

    return kernel_sum


def compute_mmd(posterior_z, prior_z):
    pos_z_batch = posterior_z.size(0)
    prior_z_batch = prior_z.size(0)

    prior_z_kernel = compute_kernel(prior_z, prior_z, exclude_diag=True)
    posterior_z_kernel = compute_kernel(posterior_z, posterior_z, exclude_diag=True)
    combined_kernel = compute_kernel(prior_z, posterior_z, exclude_diag=False)

    mmd = prior_z_kernel.div(prior_z_batch*(prior_z_batch-1)) + posterior_z_kernel.div(pos_z_batch*(pos_z_batch-1)) - \
        2 * combined_kernel.mean()
    return mmd
