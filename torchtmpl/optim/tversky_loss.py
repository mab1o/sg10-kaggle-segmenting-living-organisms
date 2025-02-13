import segmentation_models_pytorch as smp
import torch.nn as nn


class TverskyLoss(nn.Module):
    """Tversky Loss = 1 - (TP / (TP + alpha*FP + beta*FN)).

    alpha < beta => more penalty on false negatives (FN)
    """

    def __init__(self, config_loss):
        super().__init__()
        alpha = config_loss.get("alpha", 0.6)
        beta = config_loss.get("beta", 0.4)
        self.loss = smp.losses.TverskyLoss(
            mode="binary", alpha=alpha, beta=beta, from_logits=True
        )

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)
