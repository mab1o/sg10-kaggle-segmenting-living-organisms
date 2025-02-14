import torch
import torch.nn as nn
import logging


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss = (1 - Tversky Index) ^ gamma."""

    def __init__(self, config_loss):
        super().__init__()
        self.alpha = config_loss.get("alpha", 0.3)
        self.beta = config_loss.get("beta", 0.7)
        self.gamma = config_loss.get("gamma", 1.1)
        self.smooth = config_loss.get("smooth", 1e-6)
        self.reduction = config_loss.get("reduction", "mean")

    def forward(self, inputs, targets):
        if targets.ndimension() == 3:
            targets = targets.unsqueeze(1)

        if inputs.ndimension() == 3:
            inputs = inputs.unsqueeze(1)

        inputs = torch.sigmoid(inputs)

        tp = (inputs * targets).sum(dim=(1, 2, 3))
        fp = ((1 - targets) * inputs).sum(dim=(1, 2, 3))
        fn = (targets * (1 - inputs)).sum(dim=(1, 2, 3))

        # Ensure numerical stability
        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        tversky_index = torch.clamp(tversky_index, min=1e-4, max=1.0)

        # Compute Focal Tversky Loss safely
        focal_tversky_loss = torch.pow(
            torch.clamp(1 - tversky_index, min=1e-4), self.gamma
        )

        # Debugging check
        if (
            torch.isnan(focal_tversky_loss).any()
            or torch.isinf(focal_tversky_loss).any()
        ):
            logging.info(f"NaN/Inf detected: Tversky Index={tversky_index}")

        return (
            focal_tversky_loss.mean()
            if self.reduction == "mean"
            else focal_tversky_loss.sum()
        )
