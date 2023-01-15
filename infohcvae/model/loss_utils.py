import torch


# def compute_kernel(x1, x2, kernel_type="imq", exclude_diag: bool = None):
#     if kernel_type == "rbf":
#         result = compute_mmd_with_rbf(x1, x2)
#     elif kernel_type == "imq":
#         assert exclude_diag is not None, "IMQ kernel requires `exclude_diag` argument"
#         result = compute_mmd_with_inv_mult_quad(x1, x2, exclude_diag=exclude_diag)
#     else:
#         raise ValueError('Undefined kernel type.')
#
#     return result


def _compute_mmd_with_rbf(x1, x2):
    batch_size, h_dim = x1.size()

    norms_x = x1.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(x1, x1.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = x2.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(x2, x2.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(x1, x2.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 / scale
        res1 = torch.exp(-C * dists_x)
        res1 += torch.exp(-C * dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = torch.exp(-C * dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2

    return stats


def _compute_mmd_with_inv_mult_quad(z1, z2):
    batch_size, h_dim = z1.size()

    norms_x = z1.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_x = torch.mm(z1, z1.t())  # batch_size x batch_size
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = z2.pow(2).sum(1, keepdim=True)  # batch_size x 1
    prods_y = torch.mm(z2, z2.t())  # batch_size x batch_size
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(z1, z2.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * h_dim * 1.0 * scale
        res1 = C / (C + dists_x)
        res1 += C / (C + dists_y)

        if torch.cuda.is_available():
            res1 = (1 - torch.eye(batch_size).cuda()) * res1
        else:
            res1 = (1 - torch.eye(batch_size)) * res1

        res1 = res1.sum() / (batch_size - 1)
        res2 = C / (C + dists_c)
        res2 = res2.sum() * 2. / batch_size
        stats += res1 - res2

    return stats / batch_size


def compute_mmd(posterior_z, prior_z, kernel_type="imq"):
    # bs = posterior_z.size(0)
    #
    # prior_z_kernel = compute_kernel(prior_z, prior_z, exclude_diag=True)
    # posterior_z_kernel = compute_kernel(posterior_z, posterior_z, exclude_diag=True)
    # combined_kernel = compute_kernel(prior_z, posterior_z, exclude_diag=False)
    #
    # mmd = prior_z_kernel.mean() + posterior_z_kernel.mean() - \
    #     2 * combined_kernel.mean()
    # return mmd
    if kernel_type == "rbf":
        return _compute_mmd_with_rbf(posterior_z, prior_z)
    else:
        return _compute_mmd_with_inv_mult_quad(posterior_z, prior_z)
