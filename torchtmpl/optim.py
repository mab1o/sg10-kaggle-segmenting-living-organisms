# coding: utf-8

# External imports
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, ReduceLROnPlateau,
    CosineAnnealingLR, CosineAnnealingWarmRestarts
)
# local imports
from . import losses


def get_loss(lossname: str, config_loss, device):
    if lossname == "BCEWithLogitsLoss":
        pos_weight = torch.tensor([config_loss.get("pos_weight", 1.0)], device=device)
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    elif lossname == "TverskyLoss":
        alpha = config_loss.get("alpha", 0.6)
        beta = config_loss.get("beta", 0.4)
        return  smp.losses.TverskyLoss(mode="binary", alpha=alpha, beta=beta, from_logits=True)
        #return losses.TverskyLoss(alpha=alpha, beta=beta)
        #this loss is less efficient than the one in the smp library

    elif lossname == "FocalTverskyLoss":
        return losses.FocalTverskyLoss(
            alpha=config_loss.get("alpha", 0.3),
            beta=config_loss.get("beta", 0.7),
            gamma=config_loss.get("gamma", 1.1)
        )
    
    elif lossname == "Tversky-BCE":
        bce_loss = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([config_loss.get("pos_weight", 1.0)], device=device)
        )
        tversky_loss = losses.TverskyLoss(
            alpha=config_loss.get("alpha", 0.3), 
            beta =config_loss.get("beta", 0.7)
        )
        return lambda pred,target : 0.8 * tversky_loss(pred, target) + 0.2 * bce_loss(pred, target)
    
    else:
        return eval(f"nn.{lossname}()")


def get_optimizer(cfg, params):
    params_dict = cfg["params"]
    exec(f"global optim; optim = torch.optim.{cfg['algo']}(params, **params_dict)")
    return optim


def get_scheduler(optimizer, config_scheduler):
    """Retourne un scheduler PyTorch basé sur la config YAML."""
    schedulers = {
        "steplr": StepLR,
        "multisteplr": MultiStepLR,
        "exponentiallr": ExponentialLR,
        "cosineannealinglr": CosineAnnealingLR,
        "cosineannealingwarmrestarts": CosineAnnealingWarmRestarts
    }

    scheduler_name = config_scheduler.get("class", "").lower()
    scheduler_params = config_scheduler.get("params", {})

    if scheduler_name in schedulers:
        return schedulers[scheduler_name](optimizer, **scheduler_params)
    
    raise ValueError(f"Scheduler {scheduler_name} not recognized.")