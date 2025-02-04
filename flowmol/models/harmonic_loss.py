import torch
import torch.nn as nn
import torch.nn.functional as F

class HarmonicLoss(nn.Module):

    """
    Harmonic loss for categorical data as proposed by https://arxiv.org/abs/2502.01628
    
    
    Must be used in conjunction with flowmol.models.distlayer.DistLayer
    """

    def __init__(self, reduction: str = 'mean', ignore_index=None):
        super(HarmonicLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index

        if self.reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction: {self.reduction}")

    def forward(self, logits, target):
        """
        Computes the loss given unnormalized class probabilities (logits) and target labels.
        
        Args:
            logits (torch.Tensor): Raw logits of shape (batch_size, num_classes).
            target (torch.Tensor): Target labels of shape (batch_size,).
        
        Returns:
            torch.Tensor: Scalar loss value.
        """
        # Convert logits to probabilities if needed (optional, depends on loss design)
        probs = logits/torch.sum(logits, dim=-1, keepdim=True)

        # create a mask of things that are the ignore index
        if self.ignore_index is not None:
            ignore_mask = target == self.ignore_index 
            target[ignore_mask] = 0 # need to do this so we dont get index error in target_probs computation

        # Compute loss (Example: Negative Log-Likelihood)
        # Gather the probabilities corresponding to the target labels
        target_probs = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        
        # Compute loss (Example: Negative Log-Likelihood)
        loss = -1 * target_probs

        if self.ignore_index is not None:
            # Zero out the loss for the ignored index
            loss[ignore_mask] = 0

        # apply reduction
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")

        return loss