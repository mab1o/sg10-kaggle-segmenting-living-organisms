# External imports
import segmentation_models_pytorch as smp


def Unet(cfg, input_size, num_classes):
    return smp.Unet(
        encoder_name=cfg["encoder"]["model_name"],
        encoder_weights="imagenet",
        in_channels=input_size[0],
        classes=num_classes,
        decoder_attention_type="scse",
    )


def UnetPlusPlus(cfg, input_size, num_classes):
    return smp.UnetPlusPlus(
        encoder_name=cfg["encoder"]["model_name"],
        encoder_weights="imagenet",
        in_channels=input_size[0],
        classes=num_classes,
        decoder_attention_type="scse",
    )


def DeepLabV3(cfg, input_size, num_classes):
    return smp.DeepLabV3(
        encoder_name=cfg["encoder"]["model_name"],
        encoder_weights="imagenet",
        in_channels=input_size[0],
        classes=num_classes,
    )


def DeepLabV3Plus(cfg, input_size, num_classes):
    return smp.DeepLabV3Plus(
        encoder_name=cfg["encoder"]["model_name"],
        encoder_weights="imagenet",
        in_channels=input_size[0],
        classes=num_classes,
    )


def FPN(cfg, input_size, num_classes):
    return smp.FPN(
        encoder_name=cfg["encoder"]["model_name"],
        encoder_weights="imagenet",
        in_channels=input_size[0],
        classes=num_classes,
    )


def PAN(cfg, input_size, num_classes):
    return smp.PAN(
        encoder_name=cfg["encoder"]["model_name"],
        encoder_weights="imagenet",
        in_channels=input_size[0],
        classes=num_classes,
    )


def PSPNet(cfg, input_size, num_classes):
    return smp.PSPNet(
        encoder_name=cfg["encoder"]["model_name"],
        encoder_weights="imagenet",
        in_channels=input_size[0],
        classes=num_classes,
    )


def Linknet(cfg, input_size, num_classes):
    return smp.Linknet(
        encoder_name=cfg["encoder"]["model_name"],
        encoder_weights="imagenet",
        in_channels=input_size[0],
        classes=num_classes,
    )


def Segformer(cfg, input_size, num_classes):
    return smp.Segformer(
        encoder_name=cfg["encoder"]["model_name"],
        encoder_weights="imagenet",
        in_channels=input_size[0],
        classes=num_classes,
        decoder_segmentation_channels=256,
    )
