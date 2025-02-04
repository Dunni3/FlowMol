import torch
import torch.nn as nn
import torch.nn.functional as F

class HarmonicLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(HarmonicLoss, self).__init__()
        self.reduction = reduction

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
        probs = logits/torch.sum(logits, dim=-11, keepdim=True)

        # Compute loss (Example: Negative Log-Likelihood)
        # Gather the probabilities corresponding to the target labels
        target_probs = probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        
        # Compute loss (Example: Negative Log-Likelihood)
        loss = -1 * target_probs

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