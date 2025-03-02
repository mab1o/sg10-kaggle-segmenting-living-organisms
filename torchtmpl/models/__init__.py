"""Provide utilities for building segmentation models.

It integrates both custom models defined within the package and standard models
from the segmentation_models_pytorch library. The `build_model` function constructs
a model based on a configuration dictionary, input size, and number of classes.

"""

# Local imports

# Custom Models
from .cnn_models import vanilla_cnn
from .efficientnet_b3_segmentation import EfficientNetB3Segmentation
from .unet import UNet
from .torch_unet import UNet_custom
from .torch_segformer import SegFormer_custom

# Standard Models from smp
from .smp_models import *


def build_model(cfg, input_size, num_classes):
    return eval(f"{cfg['class']}(cfg, input_size, num_classes)")
