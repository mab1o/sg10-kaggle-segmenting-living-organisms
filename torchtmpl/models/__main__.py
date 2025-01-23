# coding: utf-8

# External imports
import torch
import logging

# Local imports
from . import build_model


def test_linear():
    cfg = {"class": "Linear"}
    input_size = (3, 32, 32)
    batch_size = 64
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")


def test_cnn():
    cfg = {"class": "VanillaCNN", "num_layers": 4}
    input_size = (3, 32, 32)
    batch_size = 64
    num_classes = 18
    model = build_model(cfg, input_size, num_classes)

    input_tensor = torch.randn(batch_size, *input_size)
    output = model(input_tensor)
    print(f"Output tensor of size : {output.shape}")

def test_unet():
    logging.info("Testing UNet")
    cin = 1
    input_size = (cin, 32, 32)
    num_classes = 2
    X = torch.zeros((1, *input_size))
    cfg = {"class": "UNet", "encoder": {"model_name": "resnet18"}}
    model = build_model(cfg,input_size,num_classes,)

    model.eval()
    y = model(X)
    print(f"Output shape : {y.shape}")
    print(y[0, :, 1, :].shape)
    assert y.shape == (1, num_classes, input_size[1], input_size[2])



def test_efficientnet_b3_segmentation():
    import torch
    from timm import create_model
    from efficientnet_b3_segmentation import EfficientNetB3Segmentation  # Ensure your class is named and imported correctly

    logging.info("Testing EfficientNet-B3 Segmentation")
    input_channels = 1  # Number of input channels (e.g., grayscale = 1, RGB = 3)
    input_size = (input_channels, 256, 256)  # Example input size (C, H, W)
    num_classes = 1  # Binary segmentation (adjust if needed)

    # Instantiate the model
    model = EfficientNetB3Segmentation(input_channels=input_channels, num_classes=num_classes)

    # Set the model to evaluation mode
    model.eval()

    # Generate a dummy input tensor
    batch_size = 1  # Single image for testing
    X = torch.randn(batch_size, *input_size)  # Example input with random values

    # Forward pass
    with torch.no_grad():
        y = model(X)

    # Print output shape
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")

    # Assert that the output shape matches the expected size
    assert y.shape == (batch_size, num_classes, input_size[1], input_size[2]), \
        "Output shape does not match expected shape!"

if __name__ == "__main__":
    # test_linear()
    # test_cnn()
    #test_unet()
    test_efficientnet_b3_segmentation()
