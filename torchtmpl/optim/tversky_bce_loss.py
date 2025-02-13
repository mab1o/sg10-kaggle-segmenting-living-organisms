import torch
import torch.nn as nn

from .bce_loss import BCEWithLogitsLoss
from .tversky_loss import TverskyLoss


class Tversky_BCE(nn.Module):
    def __init__(self, config_loss):
        super().__init__()
        self.bce_loss = BCEWithLogitsLoss(config_loss)
        self.tversky_loss = TverskyLoss(config_loss)
        self.loss = lambda inputs, targets: 0.8 * self.tversky_loss(
            inputs, targets
        ) + 0.2 * self.bce_loss(inputs, targets)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)
