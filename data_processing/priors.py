from torch.distributions import Exponential
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn.functional import softmax

@torch.no_grad()
def compute_ot_prior(dst_dict, pos_prior_std: float = 4.0, rotate_positions: bool = False):
    prior_dict = {}
    cat_features = ['a', 'c', 'e']

    for feat in dst_dict.keys():

        # get destination features (t=1)
        dst_feat = dst_dict[feat]

        # sample prior
        if feat in cat_features:
            prior_feat = uniform_simplex_prior(dst_feat.shape[0], d=dst_feat.shape[1])
        else:
            assert feat == 'x', "there is a non-categorical feature that is not position, this is not supported yet."
            prior_feat = torch.randn(dst_feat.shape) * pos_prior_std
            prior_feat = prior_feat - prior_feat.mean(dim=0, keepdim=True)

        # solve assignment problem
        cost_mat = torch.cdist(dst_feat, prior_feat, p=2)
        _, prior_idx = linear_sum_assignment(cost_mat)

        # reorder prior to according to optimal assignment
        prior_feat = prior_feat[prior_idx]

        if feat == 'x' and rotate_positions:
            # perform rigid alignment
            prior_feat = rigid_alignment(prior_feat, dst_feat, pre_centered=True)

        prior_dict[feat] = prior_feat

    return prior_dict

def biased_simplex_prior(n_samples, zero_order_weight: float = 0.75, std: float = 0.2, d=5):
    """
    Generate samples from a simplex which are biased towards the 0-indexed category.
    """
    non_zero_weight = (1 - zero_order_weight) / (d - 1)
    mu = torch.ones(d)*non_zero_weight
    mu[0] = zero_order_weight
    simplex_sample = mu.unsqueeze(0) + torch.randn(n_samples, d)*std
    simplex_sample = softmax(simplex_sample/(1/d), dim=1)
    return simplex_sample

def uniform_simplex_prior(n_samples, d=5):
    """
    Generate samples from a uniform distribution on a simplex.
    """
    exp_dist = Exponential(torch.tensor(1.0))
    sample = exp_dist.sample((n_samples, d))
    sample = sample / sample.sum(dim=1, keepdim=True)
    return sample


@torch.no_grad()
def rigid_alignment(x_0, x_1, pre_centered=False):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    Alignment of two point clouds using the Kabsch algorithm.
    Based on: https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
    """
    d = x_0.shape[1]
    assert x_0.shape == x_1.shape, "x_0 and x_1 must have the same shape"

    # remove COM from data and record initial COM
    if pre_centered:
        x_0_mean = torch.zeros(1, d)
        x_1_mean = torch.zeros(1, d)
        x_0_c = x_0
        x_1_c = x_1
    else:
        x_0_mean = x_0.mean(dim=0, keepdim=True)
        x_1_mean = x_1.mean(dim=0, keepdim=True)
        x_0_c = x_0 - x_0_mean
        x_1_c = x_1 - x_1_mean

    # Covariance matrix
    H = x_0_c.T.mm(x_1_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    if pre_centered:
        t = torch.zeros(1, d)
    else:
        t = x_1_mean - R.mm(x_0_mean.T).T # has shape (1, D)

    # apply rotation to x_0_c
    x_0_aligned = x_0_c.mm(R.T)

    # move x_0_aligned to its original frame
    x_0_aligned = x_0_aligned + x_0_mean

    # apply the translation
    x_0_aligned = x_0_aligned + t

    return x_0_aligned