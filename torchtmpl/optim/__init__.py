# External imports
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as scheduler

# Local imports
from .bce_loss import BCEWithLogitsLoss
from .tversky_loss import TverskyLoss
from .focal_tversky_loss import FocalTverskyLoss
from .tversky_bce_loss import Tversky_BCE


def get_loss(lossname: str, config_loss):
    custom_loss = {
        "BCEWithLogitsLoss": BCEWithLogitsLoss,
        "TverskyLoss": TverskyLoss,
        "FocalTverskyLoss": FocalTverskyLoss,
        "Tversky_BCE": Tversky_BCE,
    }

    if lossname in custom_loss:
        return custom_loss[lossname](config_loss)
    elif hasattr(nn, lossname):
        return getattr(nn, lossname)()

    raise ValueError(f"Loss function {lossname} not recognized.")


def get_optimizer(cfg, params):
    optimizer_class = getattr(torch.optim, cfg["algo"], None)
    if optimizer_class is None:
        raise ValueError(f"Optimizer {cfg['algo']} not recognized.")
    return optimizer_class(params, **cfg["params"])


def get_scheduler(optimizer, config_scheduler):
    scheduler_class = getattr(scheduler, config_scheduler.get("class", ""), None)
    if scheduler_class is None:
        raise ValueError(
            f"Scheduler {config_scheduler.get('class', '')} not recognized."
        )
    return scheduler_class(optimizer, **config_scheduler.get("params", {}))
