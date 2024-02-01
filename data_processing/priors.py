from torch.distributions import Exponential
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn.functional import softmax

def compute_ot_prior(dst_dict, pos_prior_std: float = 4.0):
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
        prior_dict[feat] = prior_feat[prior_idx]

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