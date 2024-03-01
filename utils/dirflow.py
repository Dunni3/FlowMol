import torch
import numpy as np
import scipy

class DirichletConditionalFlow:
    def __init__(self, K=20, alpha_min=1, alpha_max=100, alpha_spacing=0.01):
        self.alphas = np.arange(alpha_min, alpha_max + alpha_spacing, alpha_spacing)
        self.beta_cdfs = []
        self.bs = np.linspace(0, 1, 1000)
        for alph in self.alphas:
            self.beta_cdfs.append(scipy.special.betainc(alph, K-1, self.bs))
        self.beta_cdfs = np.array(self.beta_cdfs)
        self.beta_cdfs_derivative = np.diff(self.beta_cdfs, axis=0) / alpha_spacing
        self.K = K

    def c_factor(self, bs, alpha):
        out1 = scipy.special.beta(alpha, self.K - 1)
        out2 = np.where(bs < 1, out1 / ((1 - bs) ** (self.K - 1)), 0)
        out = np.where(bs > 0, out2 / (bs ** (alpha - 1)), 0)
        I_func = self.beta_cdfs_derivative[np.argmin(np.abs(alpha - self.alphas))]
        interp = -np.interp(bs, self.bs, I_func)
        final = interp * out
        return final
    
def sample_cond_prob_path(args, seq, alphabet_size):
    B, L = seq.shape
    seq_one_hot = torch.nn.functional.one_hot(seq, num_classes=alphabet_size)
    if args.mode == 'dirichlet':
        alphas = torch.from_numpy(1 + scipy.stats.expon().rvs(size=B) * args.alpha_scale).to(seq.device).float()
        if args.fix_alpha:
            alphas = torch.ones(B, device=seq.device) * args.fix_alpha
        alphas_ = torch.ones(B, L, alphabet_size, device=seq.device)
        alphas_ = alphas_ + seq_one_hot * (alphas[:,None,None] - 1)
        xt = torch.distributions.Dirichlet(alphas_).sample()
    elif args.mode == 'distill':
        alphas = torch.zeros(B, device=seq.device)
        xt = torch.distributions.Dirichlet(torch.ones(B, L, alphabet_size, device=seq.device)).sample()
    elif args.mode == 'riemannian':
        t = torch.rand(B, device=seq.device)
        dirichlet = torch.distributions.Dirichlet(torch.ones(alphabet_size, device=seq.device))
        x0 = dirichlet.sample((B,L))
        x1 = seq_one_hot
        xt = t[:,None,None] * x1 + (1 - t[:,None,None]) * x0
        alphas = t
    elif args.mode == 'ardm' or args.mode == 'lrar':
        mask_prob = torch.rand(1, device=seq.device)
        mask = torch.rand(seq.shape, device=seq.device) < mask_prob
        if args.mode == 'lrar': mask = ~(torch.arange(L, device=seq.device) < (1-mask_prob) * L)
        xt = torch.where(mask, alphabet_size, seq) # mask token index
        xt = torch.nn.functional.one_hot(xt, num_classes=alphabet_size + 1).float() # plus one to include index for mask token
        alphas = mask_prob.expand(B)
    return xt, alphas