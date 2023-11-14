import torch
import torch.nn as nn
from typing import Dict, Tuple

class InterpolantScheduler(nn.Module):

    def __init__(self, canonical_feat_order: str, schedule_type: str = 'cosine', cosine_params: dict = {}):
        super().__init__()

        self.feats = canonical_feat_order
        self.n_feats = len(self.feats)
        self.schedule_type = schedule_type

        if schedule_type == 'cosine':
            self.alpha_t = self.cosine_alpha_t
            self.alpha_t_prime = self.cosine_alpha_t_prime
        elif schedule_type == 'linear':
            self.alpha_t = self.linear_alpha_t
            self.alpha_t_prime = self.linear_alpha_t_prime
        else:
            raise NotImplementedError(f'unsupported schedule_type: {schedule_type}')
        
        if schedule_type == 'cosine':
            for feat in self.feats:
                if feat not in cosine_params:
                    raise ValueError(f'must specify cosine_params for feature {feat}')
                
        if schedule_type == 'cosine':
            cosine_params = [ cosine_params[feat] for feat in self.feats ]
            cosine_params = torch.tensor(cosine_params).unsqueeze(0)

        self.device = None
        
        self.cosine_params = cosine_params

    def interpolant_weights(self, t: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns the weights for x_0 and x_1 in the interpolation between x_0 and x_1.
        """

        if self.schedule_type == 'cosine' and t.device != self.device:
            self.cosine_params = self.cosine_params.to(t.device)
            self.device = t.device

        alpha_t = self.alpha_t(t)

        weights = (1 - alpha_t, alpha_t)
        return weights
    
    def loss_weights(self, t: torch.Tensor):
        alpha_t = self.alpha_t(t)
        # alpha_t_prime = self.alpha_t_prime(t)
        # weights = alpha_t_prime/(1 - alpha_t + 1e-5)
        weights = alpha_t/(1 - alpha_t)

        # clamp the weights with a minimum of 0.05 and a maximum of 1.5
        weights = torch.clamp(weights, min=0.05, max=1.5)
        return weights

    def cosine_alpha_t(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        nu = self.cosine_params
        t = t.unsqueeze(-1)
        alpha_t = 1 - torch.cos(torch.pi*0.5*torch.pow(t, nu)).square()
        return alpha_t
    
    def cosine_alpha_t_prime(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        nu = self.cosine_params
        t = t.unsqueeze(-1)
        sin_input = torch.pi*torch.pow(t, nu)
        alpha_t_prime = torch.pi*0.5*torch.sin(sin_input)*nu*torch.pow(t, nu-1)
        return alpha_t_prime
    
    def linear_alpha_t(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        alpha_t = t.unsqueeze(-1).repeat(1, self.n_feats)
        return alpha_t
    
    def linear_alpha_t_prime(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        alpha_t_prime = torch.ones_like(t).unsqueeze(-1).repeat(1, self.n_feats)
        return alpha_t_prime