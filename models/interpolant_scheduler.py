import torch
import torch.nn as nn
from typing import Dict, Tuple

class InterpolantScheduler(nn.Module):

    def __init__(self, schedule_type: str = 'cosine', cosine_params: dict = {}):
        super().__init__()

        self.feats = ['x', 'a', 'c', 'e']

        if schedule_type == 'cosine':
            self.alpha_t = self.cosine_alpha_t
        elif schedule_type == 'linear':
            self.alpha_t = self.linear_alpha_t
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

    def cosine_alpha_t(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        alpha_t = {}
        for feat in self.feats:
            nu = self.cosine_params[feat]
            alpha_t[feat] = 1 - torch.cos(torch.pi*0.5*torch.pow(t, nu)).square()

        return alpha_t
    
    def linear_alpha_t(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:

        alpha_t = {}
        for feat in self.feats:
            alpha_t[feat] = t

        return alpha_t