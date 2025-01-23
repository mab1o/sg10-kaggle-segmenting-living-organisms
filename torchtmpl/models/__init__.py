# coding: utf-8

# External imports
import torch

# Local imports
from .base_models import *
from .cnn_models import *
from .unet import *
from .efficientnet_b3_segmentation import *



def build_model(cfg, input_size, num_classes):
    if cfg['class'] == 'EfficientNetB3Segmentation':
        return EfficientNetB3Segmentation(
            input_channels=input_size[0], 
            num_classes=num_classes
        )
    # Handle other models
    return eval(f"{cfg['class']}(cfg, input_size, num_classes)")