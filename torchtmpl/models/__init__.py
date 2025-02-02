# coding: utf-8

# External imports
import torch

# Local imports
from .base_models import *
from .cnn_models import *
from .unet import *
from .efficientnet_b3_segmentation import *
#from .unet_convnext import *
import segmentation_models_pytorch as smp


def build_model(cfg, input_size, num_classes):
    if cfg['class'] == 'EfficientNetB3Segmentation':
        return EfficientNetB3Segmentation(
            input_channels=input_size[0], 
            num_classes=num_classes
        )
    elif cfg['class'] == 'UNet' and 'regnety_032' in cfg['encoder']['model_name']:
            return smp.Unet(
                encoder_name="timm-regnety_032",  # RegNetY-3.2GF
                encoder_weights="imagenet",       # Poids pré-entraînés
                in_channels=1,                    # Images en niveaux de gris
                classes=1                         # Segmentation binaire
            )
    elif cfg['class'] == 'UNet++' and 'regnety_032' in cfg['encoder']['model_name']:
        return smp.UnetPlusPlus(
            encoder_name="timm-regnety_032",  # RegNetY-3.2GF
            encoder_weights="imagenet",       # Poids pré-entraînés
            in_channels=1,                    # Images en niveaux de gris
            classes=1                         # Segmentation binaire
        )
    elif cfg['class'] == 'DeepLabV3Plus' :
        return smp.DeepLabV3Plus(
            encoder_name=cfg['encoder']['model_name'],  # SegFormer, better than CNNs
            encoder_weights="imagenet",
            in_channels=1,
            classes=1
        )
    return eval(f"{cfg['class']}(cfg, input_size, num_classes)")