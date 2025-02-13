import torch
import torch.nn as nn


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, config_loss):
        super().__init__()
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda") if use_cuda else torch.device("cpu")

        pos_weight = torch.tensor([config_loss.get("pos_weight", 1.0)], device=device)
        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, inputs, targets):
        return self.loss(inputs, targets)
