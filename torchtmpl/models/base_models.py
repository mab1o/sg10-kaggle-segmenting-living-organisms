# Standard imports
import operator
from functools import reduce

# External imports
import torch.nn as nn


def linear(cfg, input_size, num_classes):
    """cfg: a dictionnary with possibly some parameters.

    input_size: (C, H, W) input size tensor
    num_classes: int
    """
    layers = [
        nn.Flatten(start_dim=1),
        nn.Linear(reduce(operator.mul, input_size, 1), num_classes),
    ]
    return nn.Sequential(*layers)
