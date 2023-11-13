import torch
import torch.nn as nn
from typing import Dict, Tuple

class InterpolantScheduler(nn.Module):

    def __init__(self, schedule_type: str = 'cosine', cosine_params: dict = {}):
        super().__init__()

        self.feats = ['x', 'a', 'c', 'e']

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
                
                
        
        self.cosine_params = cosine_params    

    def interpolant_weights(self, t: torch.Tensor) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Returns the weights for x_0 and x_1 in the interpolation between x_0 and x_1.
        """
        alpha_t = self.alpha_t(t)
        weights = {}
        for feat in alpha_t:
            weights[feat] = (1-alpha_t[feat], alpha_t[feat])
        return weights
    
    def loss_weights(self, t: torch.Tensor):
        alpha_t = self.alpha_t(t)
        alpha_t_prime = self.alpha_t_prime(t)
        weights = {}
        for feat in self.feats:
            weights[feat] = alpha_t_prime[feat]/(1 - alpha_t[feat] + 1e-5)
        return weights

    def cosine_alpha_t(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        alpha_t = {}
        for feat in self.feats:
            nu = self.cosine_params[feat]
            alpha_t[feat] = 1 - torch.cos(torch.pi*0.5*torch.pow(t, nu)).square()

        return alpha_t
    
    def cosine_alpha_t_prime(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        alpha_t_prime = {}
        for feat in self.feats:
            nu = self.cosine_params[feat]
            cos_input = torch.pi*torch.power(t, nu)
            alpha_t_prime[feat] = torch.pi*0.5*torch.cos(cos_input)*nu*torch.pow(t, nu-1)
            
        return alpha_t_prime
    
    def linear_alpha_t(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:

        alpha_t = {}
        for feat in self.feats:
            alpha_t[feat] = t

        return alpha_t
    
    def linear_alpha_t_prime(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        alpha_t_prime = {}
        for feat in self.feats:
            alpha_t_prime[feat] = torch.ones_like(t)
            
        return alpha_t_prime