import torch
import torch.nn as nn
from typing import Dict, Tuple, Union

class InterpolantScheduler(nn.Module):

    supported_schedule_types = ['cosine', 'linear']

    def __init__(self, canonical_feat_order: str, schedule_type: Union[str, Dict[str, str]] = 'cosine', cosine_params: dict = {}):
        super().__init__()

        self.feats = canonical_feat_order
        self.n_feats = len(self.feats)

        # check that schedule_type is a string or a dictionary
        if not isinstance(schedule_type, (str, dict)):
            raise ValueError('schedule_type must be a string or a dictionary')
        
        # if it is a string, assign the same schedule_type to all features
        if isinstance(schedule_type, str):
            if schedule_type not in self.supported_schedule_types:
                raise ValueError(f'unsupported schedule_type: {schedule_type}')
            self.schedule_dict = {
                feat: schedule_type for feat in self.feats
            }
        else:
            # schedule_type is a dictionary specifying the schedule_type for each feature
            for feat in self.feats:
                if feat not in schedule_type:
                    raise ValueError(f'must specify schedule_type for feature {feat}')

            self.schedule_dict = schedule_type 

        # if schedule_type == 'cosine':
        #     self.alpha_t = self.cosine_alpha_t
        #     self.alpha_t_prime = self.cosine_alpha_t_prime
        # elif schedule_type == 'linear':
        #     self.alpha_t = self.linear_alpha_t
        #     self.alpha_t_prime = self.linear_alpha_t_prime
        # else:
        #     raise NotImplementedError(f'unsupported schedule_type: {schedule_type}')
            

        # for features which have a cosine schedule, check that the parameter "nu" is provided
        for feat, schedule_type in self.schedule_dict.items():
            if schedule_type == 'cosine' and feat not in cosine_params:
                raise ValueError(f'must specify cosine_params for feature {feat}')
    
        # get a list of unique schedule types which are used
        self.schedule_types = list(set( self.schedule_dict.values() ))

        # if we are using a cosine schedule, convert all of the cosine_params to torch tensors
        if 'cosine' in self.schedule_types:
            for feat in cosine_params:
                cosine_params[feat] = torch.tensor(cosine_params[feat]).unsqueeze(0)
        
        # save the cosine_params as an attribute
        self.cosine_params = cosine_params

        self.device = None

        self.clamp_t = True

        

    def update_device(self, t):
        if 'cosine' in self.schedule_types and t.device != self.device:
            for key in self.cosine_params:
                self.cosine_params[key] = self.cosine_params[key].to(t.device)
            self.device = t.device

    def interpolant_weights(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns the weights for x_0 and x_1 in the interpolation between x_0 and x_1.
        """
        # t has shape (n_timepoints,)
        # returns a tuple of 2 tensors of shape (n_timepoints, n_feats)
        # the tensor at index 0 is the weight for x_0
        # the tensor at index 1 is the weight for x_1

        self.update_device(t)

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
    
    def alpha_t(self, t: torch.Tensor) -> torch.Tensor:

        self.update_device(t)

        per_feat_alpha = []
        for feat in self.feats:
            schedule_type = self.schedule_dict[feat]
            if schedule_type == 'cosine':
                alpha_t = self.cosine_alpha_t(t, nu=self.cosine_params[feat])
            elif schedule_type == 'linear':
                alpha_t = self.linear_alpha_t(t)
            
            per_feat_alpha.append(alpha_t)

        alpha_t = torch.cat(per_feat_alpha, dim=1)
        return alpha_t
    
    def alpha_t_prime(self, t: torch.Tensor) -> torch.Tensor:
        self.update_device(t)

        per_feat_alpha_prime = []
        for feat in self.feats:
            schedule_type = self.schedule_dict[feat]
            if schedule_type == 'cosine':
                alpha_t_prime = self.cosine_alpha_t_prime(t, nu=self.cosine_params[feat])
            elif schedule_type == 'linear':
                alpha_t_prime = self.linear_alpha_t_prime(t)
            
            per_feat_alpha_prime.append(alpha_t_prime)

        alpha_t_prime = torch.cat(per_feat_alpha_prime, dim=1)
        return alpha_t_prime


    def cosine_alpha_t(self, t: torch.Tensor, nu: torch.Tensor) -> Dict[str, torch.Tensor]:
        # t has shape (n_timepoints,)
        # alpha_t has shape (n_timepoints, n_feats) containing the alpha_t for each feature
        t = t.unsqueeze(-1)
        alpha_t = 1 - torch.cos(torch.pi*0.5*torch.pow(t, nu)).square()
        return alpha_t
    
    def cosine_alpha_t_prime(self, t: torch.Tensor, nu: torch.Tensor) -> torch.Tensor:

        if self.clamp_t:
            t = torch.clamp_(t, min=1e-9)

        t = t.unsqueeze(-1)
        sin_input = torch.pi*torch.pow(t, nu)
        alpha_t_prime = torch.pi*0.5*torch.sin(sin_input)*nu*torch.pow(t, nu-1)
        return alpha_t_prime
    
    def linear_alpha_t(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        alpha_t = t.unsqueeze(-1)
        return alpha_t
    
    def linear_alpha_t_prime(self, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        alpha_t_prime = torch.ones_like(t).unsqueeze(-1)
        return alpha_t_prime