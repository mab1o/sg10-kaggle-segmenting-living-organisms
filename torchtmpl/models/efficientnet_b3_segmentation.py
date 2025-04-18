import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class EfficientNetB3Segmentation(nn.Module):
    def __init__(self, cfg, input_size, num_classes):
        super().__init__()

        # Args normalised with other models
        cfg = None
        input_channels = input_size[0]

        # Load EfficientNet-B3 encoder from timm
        self.encoder = timm.create_model(
            "efficientnet_b3",
            pretrained=True,
            features_only=True,
            in_chans=input_channels,
        )

        # Decoder for segmentation
        self.decoder = nn.Conv2d(
            in_channels=self.encoder.feature_info[-1][
                "num_chs"
            ],  # Last feature map channels
            out_channels=num_classes,
            kernel_size=1,
        )

    def forward(self, x):
        # Extract features from the encoder
        features = self.encoder(x)
        # Decode to the output shape
        output = self.decoder(features[-1])
        # Upsample to match the target size
        return F.interpolate(
            output, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False
        )

    # return binary prediction mask
    def predict(self, x, threshold=0.5):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            binary_mask = (probs > threshold).long()
            return binary_mask.squeeze(1).squeeze(0)

    # return probability prediction mask
    # we return a probability and not a binary prediction to be able to print the heatmap of probability.
    def predict_probs(self, x):
        """Predict method returning logits (before sigmoid) for visualization or evaluation."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return probs
