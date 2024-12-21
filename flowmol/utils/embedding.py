import torch
import torch.nn.functional as F
import math

def get_time_embedding(timesteps, embedding_dim=256, max_positions=1000):
    # Code from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    assert len(timesteps.shape) == 1
    timesteps = timesteps * max_positions
    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

def _rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    device = D.device
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def rbf_twoscale(D, D_min=0., D_max=10, D_count=32, dividing_point: float = 3.5, high_res_frac=0.6):
    
    device = D.device

    n_highres_points = int(D_count * high_res_frac)
    n_lowres_points = D_count - n_highres_points

    D_sigma_highres = (dividing_point - D_min) / n_highres_points
    D_sigma_lowres = (D_max - dividing_point) / n_lowres_points
    sigmas = [D_sigma_highres, D_sigma_lowres]

    sections = [
        torch.linspace(D_min, dividing_point, n_highres_points, device=device),
        torch.linspace(dividing_point, D_max, n_lowres_points, device=device)[1:],
    ]
    rbf_embeddings = []
    for D_mu, D_sigma in zip(sections, sigmas):
        D_mu = D_mu.view([1, -1])
        D_expand = torch.unsqueeze(D, -1)
        RBF_i = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        rbf_embeddings.append(RBF_i)

    return torch.cat(rbf_embeddings, dim=-1)