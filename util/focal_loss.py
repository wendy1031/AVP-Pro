# File: util/focal_loss.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is designed to address class imbalance by down-weighting
    in-liers (easy examples) and focusing training on hard examples.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (torch.Tensor, float, or list): Weighting factor for each class. 
                                     If a float, it is applied to the positive class (class 1).
                                     If a list or tensor, it is applied to each class.
            gamma (float): The focusing parameter. Higher values mean more focus on hard examples.
            reduction (str): 'mean', 'sum' or 'none'.
        """
        super(FocalLoss, self).__init__()
        if isinstance(alpha, (float, int)):
            # If alpha is a single float, we assume it's for the positive class (1)
            # and the weight for the negative class (0) is 1-alpha.
            self.alpha = torch.tensor([1 - alpha, alpha], dtype=torch.float32)
        elif isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
            
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model of shape [N, C], where C is the number of classes.
            targets (torch.Tensor): Ground truth labels of shape [N].
        """
        # Ensure alpha is on the same device as the inputs
        if self.alpha is not None and self.alpha.device != inputs.device:
            self.alpha = self.alpha.to(inputs.device)

        # Calculate Cross-Entropy loss without reduction
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Get the probability of the correct class
        pt = torch.exp(-ce_loss)
        
        # Calculate the Focal Loss
        focal_loss = (1 - pt)**self.gamma * ce_loss
        
        # Apply alpha weighting if provided
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss