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


if __name__ == "__main__":
    # test_linear()
    # test_cnn()
    test_unet()
