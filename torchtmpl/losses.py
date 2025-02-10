import torch
import torch.nn as nn
from torch import Tensor


class TverskyLoss(nn.Module):
    """Tversky Loss = 1 - (TP / (TP + alpha*FP + beta*FN)).

    alpha < beta => more penalty on false negatives (FN)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1e-4,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Compute Tversky Loss.

        Args:
            inputs: Model logits (before sigmoid).
            targets: Ground truth (same shape as inputs, binary values).

        Returns:
            Tversky loss as a scalar tensor.

        """
        # Convert logits to probabilities
        inputs = torch.sigmoid(inputs)

        # Compute TP, FP, FN
        tp = (inputs * targets).sum(dim=(1, 2, 3))  # Sum over spatial dimensions
        fp = ((1 - targets) * inputs).sum(dim=(1, 2, 3))
        fn = (targets * (1 - inputs)).sum(dim=(1, 2, 3))

        # Tversky Index (Adding smooth in denominator for stability)
        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        # Compute loss
        loss = 1 - tversky_index

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class FocalTverskyLoss(nn.Module):
    """Focal Tversky Loss = (1 - Tversky Index) ^ gamma."""

    def __init__(self, alpha=0.3, beta=0.7, gamma=1.0, smooth=1e-6, reduction="mean"):
        super().__init__()
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
            print(f"NaN/Inf detected: Tversky Index={tversky_index}")

        return (
            focal_tversky_loss.mean()
            if self.reduction == "mean"
            else focal_tversky_loss.sum()
        )
