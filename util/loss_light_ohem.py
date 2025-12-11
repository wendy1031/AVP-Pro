# File: loss_light_ohem.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Standard InfoNCE loss.
    Calculates the loss for a batch of anchors, their corresponding positives,
    and a set of negatives.
    """
    def __init__(self, temperature=0.5, learnable_temperature=True, regularization=1e-4):
        super(ContrastiveLoss, self).__init__()
        self.regularization = regularization

        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature, dtype=torch.float32)))
        else:
            self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        """
        Args:
            anchor (torch.Tensor): Anchor embeddings, shape (batch_size, embedding_dim)
            positive (torch.Tensor): Positive embeddings, shape (batch_size, embedding_dim)
            negatives (torch.Tensor): Negative embeddings, shape (K, embedding_dim)
        """
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negatives = F.normalize(negatives, p=2, dim=1)

        l_pos = torch.einsum('nc,nc->n', [anchor, positive]).unsqueeze(-1)
        l_neg = torch.einsum('nc,kc->nk', [anchor, negatives])
        logits = torch.cat([l_pos, l_neg], dim=1)

        if isinstance(self.log_temperature, nn.Parameter):
            temperature = torch.exp(self.log_temperature)
            logits /= temperature
        else:
            logits /= self.temperature
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=anchor.device)
        loss = F.cross_entropy(logits, labels)
        
        if isinstance(self.log_temperature, nn.Parameter):
            loss += self.regularization * temperature
            
        return loss