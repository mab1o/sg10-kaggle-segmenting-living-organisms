import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Encoder(nn.Module):
    def __init__(self, channels=(3, 64, 128, 256, 512, 1024)):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            conv_block(channels[i], channels[i+1]) 
            for i in range(len(channels)-1)
        ])
        self.pool = nn.MaxPool2d(2)
    
    def forward(self, x):
        features = []
        for block in self.encoder_blocks:
            x = block(x)
            features.append(x)
            x = self.pool(x)
        return features

class Decoder(nn.Module):
    def __init__(self, channels=(1024, 512, 256, 128, 64)):
        super().__init__()
        self.channels = channels
        self.upconvs = nn.ModuleList([
            nn.ConvTranspose2d(channels[i], channels[i+1], 2, stride=2)
            for i in range(len(channels)-1)
        ])
        self.decoder_blocks = nn.ModuleList([
            conv_block(channels[i], channels[i+1])
            for i in range(len(channels)-1)
        ])
    
    def forward(self, x, encoder_features):
        for i in range(len(self.channels)-1):
            x = self.upconvs[i](x)
            enc_feat = encoder_features[-(i+2)]
            
            # Be cautious : potential size mismatches
            diffY = enc_feat.size()[2] - x.size()[2]
            diffX = enc_feat.size()[3] - x.size()[3]
            
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            
            x = torch.cat([x, enc_feat], dim=1)
            x = self.decoder_blocks[i](x)
        return x

class UNet(nn.Module):
    """
    Simple UNet implementation for plankton segmentation
    """
    def __init__(self, config=None):
        super().__init__()
        
        self.config = {
            'in_channels': 1,
            'out_channels': 1,
            'encoder_channels': [1, 64, 128, 256, 512, 1024],
            'dropout': 0.2
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Extract config values
        in_channels = self.config['in_channels']
        out_channels = self.config['out_channels']
        encoder_channels = self.config['encoder_channels']
        
        # Make sure first encoder channel matches in_channels
        encoder_channels[0] = in_channels
        self.encoder = Encoder(encoder_channels)
        
        # Create decoder with reversed channels (except last one)
        decoder_channels = encoder_channels[-1:0:-1]
        self.decoder = Decoder(decoder_channels)
        
        # Final convolution to get the desired number of output channels
        self.final_conv = nn.Conv2d(encoder_channels[1], out_channels, kernel_size=1)
        
        self.dropout = nn.Dropout2d(self.config['dropout'])
    
    def forward(self, x):
        encoder_features = self.encoder(x)
        
        # dropout at bottleneck
        bottleneck = encoder_features[-1]
        bottleneck = self.dropout(bottleneck)
        
        x = self.decoder(bottleneck, encoder_features)
        
        x = self.final_conv(x)
        
        return x
    
    def predict(self, x, threshold=0.5):
        """Generate binary prediction"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()

# Simple test to check the dimensions 
def test_unet():
    config = {
        'in_channels': 1,  
        'out_channels': 1, # 
        'encoder_channels': [1, 64, 128, 256, 512],  
        'dropout': 0.2
    }
    
    model = UNet()
    print(f"Model created with configuration: {config}")
    
    # dummy input
    x = torch.randn(2, 1, 256, 256)
    
    output = model(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    prediction = model.predict(x)
    print(f"Binary prediction shape: {prediction.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    return model

if __name__ == "__main__":

    test_unet()