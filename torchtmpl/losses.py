import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    """
    Tversky Loss = 1 - (TP / (TP + alpha*FP + beta*FN))
    alpha < beta => plus de pénalité sur les FN
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-4, reduction='mean'):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Convertit logits en probabilités
        inputs = torch.sigmoid(inputs)

        # Calcul TP, FP, FN
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()

        # Tversky Index
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        loss = 1 - tversky_index
        
        return loss if self.reduction == 'mean' else loss


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss = (1 - Tversky Index) ^ gamma
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.0, smooth=1e-6, reduction='mean'):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs, targets):
        if targets.ndimension() == 3:
            targets = targets.unsqueeze(1)

        inputs = torch.sigmoid(inputs)

        tp = (inputs * targets).sum(dim=(1, 2, 3))
        fp = ((1 - targets) * inputs).sum(dim=(1, 2, 3))
        fn = (targets * (1 - inputs)).sum(dim=(1, 2, 3))

        # Ensure numerical stability
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        tversky_index = torch.clamp(tversky_index, min=1e-4, max=1.0)

        # Compute Focal Tversky Loss safely
        focal_tversky_loss = torch.pow(torch.clamp(1 - tversky_index, min=1e-4), self.gamma)

        # Debugging check
        if torch.isnan(focal_tversky_loss).any() or torch.isinf(focal_tversky_loss).any():
            print(f"NaN/Inf detected: Tversky Index={tversky_index}")

        return focal_tversky_loss.mean() if self.reduction == 'mean' else focal_tversky_loss.sum()
