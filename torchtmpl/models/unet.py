# coding: utf-8

# Standard imports

# External imports
import torch
import torch.nn as nn
import timm


def conv_relu_bn(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
        nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


class TimmEncoder(nn.Module):
    def __init__(self, cin, model_name):
        super().__init__()
        self.model = timm.create_model(
            model_name=model_name, in_chans=cin, pretrained=True
        )

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)

        f1 = self.model.layer1(x)
        f2 = self.model.layer2(f1)
        f3 = self.model.layer3(f2)
        f4 = self.model.layer4(f3)

        return f4, [f1, f2, f3]


class DecoderBlock(nn.Module):
    def __init__(self, cin):
        super().__init__()
        self.conv1 = nn.Sequential(*conv_relu_bn(cin,cin))
        self.up_conv = nn.Sequential(nn.Upsample(scale_factor=2), *conv_relu_bn(cin, cin // 2))
        self.conv2 = nn.Sequential(*conv_relu_bn(cin,cin//2))

    def forward(self, x, f_encoder):
        x = self.conv1(x)
        x = self.up_conv(x)
        x = torch.cat((x, f_encoder), 1)
        out = self.conv2(x)
        return out


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        cbase = 64
        self.b1 = DecoderBlock(cin=8 * cbase)
        self.b2 = DecoderBlock(cin=4 * cbase)
        self.b3 = DecoderBlock(cin=2 * cbase)  # cout=cbase

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2), *conv_relu_bn(cbase, cbase)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2), *conv_relu_bn(cbase, cbase)
        )
        self.last_conv = nn.Sequential(*conv_relu_bn(cbase, num_classes))

    def forward(self, f4, f_encoder):
        [f1, f2, f3] = f_encoder

        x1 = self.b1(f4, f3)
        x2 = self.b2(x1, f2)
        x3 = self.b3(x2, f1)
        out = self.last_conv(self.up2(self.up1(x3)))

        return out


class UNet(nn.Module):
    """
    UNet model

    Args:
        cfg: configuration dictionary
        input_size: input image size (C, H, W)
        num_classes: number of output classes
    """

    def __init__(self, cfg, input_size, num_classes):
        super().__init__()
        cin, _, _ = input_size

        self.encoder = TimmEncoder(cin, **(cfg["encoder"]))
        self.decoder = Decoder(num_classes)

    def forward(self, X):
        out, features = self.encoder(X)
        prediction = self.decoder(out, features)
        return prediction
    
    def predict(self, X, threshold=0.5):
        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            probs = torch.sigmoid(logits)
            binary_mask = (probs > threshold).long()
            return binary_mask.squeeze(1).squeeze(0)

    def predict_probs(self, X):
        self.eval()
        with torch.no_grad():
            logits = self.forward(X)
            probs = torch.sigmoid(logits)
            return probs



def test_timm():
    x = torch.zeros((1, 3, 256, 256))
    model = timm.create_model(model_name="resnet34", pretrained=True)
    model.eval()

    x = model.conv1(x)
    x = model.bn1(x)
    x = model.act1(x)
    x = model.maxpool(x)

    f1 = model.layer1(x)
    print(f"dim of f1 : {f1.shape}")
    f2 = model.layer2(f1)
    print(f"dim of f2 : {f2.shape}")
    f3 = model.layer3(f2)
    print(f"dim of f3 : {f3.shape}")
    f4 = model.layer4(f3)
    print(f"dim of f4 : {f4.shape}")


def test_unet():
    cin = 1
    input_size = (cin, 512, 512)
    num_classes = 1
    X = torch.zeros((1, *input_size))

    cfg = {"encoder": {"model_name": "resnet18"}}

    model = UNet(cfg, input_size, num_classes)
    model.eval()
    y = model(X)

    print(f"Output shape : {y.shape}")
    assert y.shape == (1, num_classes, input_size[1], input_size[2])
    print(f"first row : {y[0, :, 0, :]}")

    y = model.predict(X)
    print(f"Predict output shape: {y.shape}")
    print(f"first row : {y[0]}")


if __name__ == "__main__":
    #test_timm()
    test_unet()
