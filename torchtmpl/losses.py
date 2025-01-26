import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    """
    Tversky Loss = 1 - (TP / (TP + alpha*FP + beta*FN))
    alpha < beta => plus de pénalité sur les FN
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-5, reduction='mean'):
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
