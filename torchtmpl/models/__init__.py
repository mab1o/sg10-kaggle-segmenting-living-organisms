# coding: utf-8

# External imports
import segmentation_models_pytorch as smp

# Local imports
from .cnn_models import VanillaCNN
from .unet import UNet
from .efficientnet_b3_segmentation import EfficientNetB3Segmentation

CUSTOM_MODELS = {
    "VanillaCNN": VanillaCNN,
    "UNet": UNet,
    "EfficientNetB3Segmentation": EfficientNetB3Segmentation
}


# List of valid segmentation models in SMP
SMP_MODELS = {
    "Unet": smp.Unet,
    "UnetPlusPlus": smp.UnetPlusPlus,
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
    "FPN": smp.FPN,
    "PAN": smp.PAN,
    "PSPNet": smp.PSPNet,
    "Linknet": smp.Linknet,
    "Segformer": smp.Segformer
}

def build_model(cfg, input_size, num_classes):
    model_class = cfg['class']
    encoder_name = cfg['encoder']['model_name']
    patch_size = cfg['patch_size'][0]

    # Validate if the model exists in SMP
    if model_class in SMP_MODELS:
        model_params = {
            "encoder_name": encoder_name,
            "encoder_weights": "imagenet",
            "in_channels": 1,  # Grayscale images
            "classes": 1  # Binary segmentation
        }
        
        # Ajout spécifique pour Unet et Unet++
        if model_class in ["Unet", "UnetPlusPlus"]:
            model_params["decoder_attention_type"] = "scse"
        
        # Ajout spécifique pour SegFormer
        if model_class == "Segformer":
            model_params["decoder_segmentation_channels"] = patch_size

        return SMP_MODELS[model_class](**model_params)

    
    if model_class == 'EfficientNetB3Segmentation':
        return EfficientNetB3Segmentation(
        input_channels=input_size[0], 
        num_classes=num_classes
    )

    if model_class in CUSTOM_MODELS:
        return CUSTOM_MODELS[model_class](cfg, input_size, num_classes)

    raise ValueError(f"Modèle inconnu : {model_class}")