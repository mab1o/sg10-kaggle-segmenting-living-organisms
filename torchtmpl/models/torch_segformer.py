import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Overlapping Patch Embedding module for SegFormer.
    """
    def __init__(self, img_size=256, patch_size=7, stride=4, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.stride = stride
        
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=stride,
            padding=(patch_size // 2)
        )
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        
        return x, H, W


class EfficientAttention(nn.Module):
    """
    Efficient Attention module for SegFormer.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.AdaptiveAvgPool2d(sr_ratio),
                nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            )
            self.norm = nn.LayerNorm(dim)
            
    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            
        k, v = kv[0], kv[1]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class MixFFN(nn.Module):
    """
    Mix Feed Forward Network module for SegFormer.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x, H, W):
        x = self.fc1(x)
        B, N, C = x.shape
        
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.
    """
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        
        return output


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio
        )
        
        drop_path_rate = drop_path if isinstance(drop_path, float) else drop_path[0]
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MixFFN(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)
        
    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class MLP(nn.Module):

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerEncoder(nn.Module):
    """
    SegFormer encoder with overlapping patch embedding + transformer blocks.
    """
    def __init__(self, img_size=256, patch_size=7, in_channels=3, embed_dims=64, 
                 num_heads=1, mlp_ratios=4, drop_rate=0., attn_drop_rate=0., 
                 drop_path_rate=0., sr_ratios=8, num_layers=2):
        super().__init__()
        self.embed_dims = embed_dims
        
        # Overlapping patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            stride=4,
            in_channels=in_channels,
            embed_dim=embed_dims
        )
        
        # Transformer blocks
        dpr = drop_path_rate
        if not isinstance(drop_path_rate, list):
            dpr = [drop_path_rate for _ in range(num_layers)]
        
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratios,
                qkv_bias=True,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                sr_ratio=sr_ratios
            )
            for i in range(num_layers)
        ])
        
    def forward(self, x):
        B = x.shape[0]
        
        x, H, W = self.patch_embed(x)
        
        for block in self.blocks:
            x = block(x, H, W)
        
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        return x


class SegFormer(nn.Module):
    """
    SegFormer: Basic implementation of arXiv:2105.15203 
    """
    def __init__(self, config=None):
        super().__init__()
        
        # presets : we define presets to limit tuning complexity
        model_presets = {
            'tiny': {
                'embed_dims': [32, 64, 128, 256],
                'depths': [2, 2, 2, 2],
                'num_heads': [1, 2, 4, 8],
            },
            'small': {
                'embed_dims': [64, 128, 256, 512],
                'depths': [3, 4, 6, 3],
                'num_heads': [1, 2, 4, 8],
            },
            'medium': {
                'embed_dims': [64, 128, 320, 512],
                'depths': [3, 4, 18, 3],
                'num_heads': [1, 2, 5, 8],
            },
            'large': {
                'embed_dims': [64, 128, 320, 640],
                'depths': [3, 8, 27, 3],
                'num_heads': [1, 2, 5, 8],
            }
        }
        
        # Base configuration with reasonable defaults
        self.config = {
            'img_size': 256,
            'in_channels': 3,
            'num_classes': 1,
            'preset': 'tiny',            # preset, I mark with 'ovr' the overriden parameters
            'embed_dims': [64, 128, 256, 512],  # ovr
            'num_heads': [1, 2, 4, 8],         # ovr
            'depths': [3, 4, 6, 3],            # ovr
            'dropout': 0.2,                    
            'drop_path': 0.1,                
            'mlp_ratio': 4,                    
            'sr_ratios': [8, 4, 2, 1]          # spatial reduction ratios
        }
        

        if config:
            self.config.update(config)
        
        # Apply preset 
        if 'preset' in self.config and self.config['preset'] in model_presets:
            preset_name = self.config['preset']
            preset_config = model_presets[preset_name]
            for key, value in preset_config.items():
                self.config[key] = value
  
        img_size = self.config['img_size']
        in_channels = self.config['in_channels']
        num_classes = self.config['num_classes']
        embed_dims = self.config['embed_dims']
        num_heads = self.config['num_heads']
        depths = self.config['depths']
        dropout = self.config['dropout']
        drop_path = self.config['drop_path']
        mlp_ratio = self.config['mlp_ratio']
        sr_ratios = self.config['sr_ratios']
        
        self.depths = depths
        self.embed_dims = embed_dims
        
        # Calculate per-stage drop path rates
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0
        
        # Encoders (multi-stage backbone)
        self.stages = nn.ModuleList()
        
        for i in range(len(depths)):
            stage_dpr = dpr[cur:cur+depths[i]]
            current_img_size = img_size // (2**i) if i > 0 else img_size
            
            # First stage uses different patch size and stride
            patch_size = 7 if i == 0 else 3
            
            self.stages.append(
                SegFormerEncoder(
                    img_size=current_img_size,
                    patch_size=patch_size,
                    in_channels=in_channels if i == 0 else embed_dims[i-1],
                    embed_dims=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratios=mlp_ratio,
                    drop_rate=dropout,
                    attn_drop_rate=dropout,
                    drop_path_rate=stage_dpr,
                    sr_ratios=sr_ratios[i],
                    num_layers=depths[i]
                )
            )
            cur += depths[i]
        
        # MLP Decoders (multi-level features)
        self.mlp_decoders = nn.ModuleList([
            MLP(input_dim=embed_dims[i], embed_dim=embed_dims[0])
            for i in range(len(embed_dims))
        ])
        
        # Segmentation head
        self.seg_head = nn.Sequential(
            nn.Conv2d(embed_dims[0] * len(embed_dims), embed_dims[0], kernel_size=1),
            nn.BatchNorm2d(embed_dims[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dims[0], num_classes, kernel_size=1)
        )
        
        # Final upsampling
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        
    def forward(self, x):
        B = x.shape[0]
        features = []
        
        # Multi-scale features from the stages
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        # MLP Decoders for all feature levels
        outs = []
        for i, feature in enumerate(features):
            # Process through MLP decoder
            decoded = self.mlp_decoders[i](feature)
            
            # Get spatial dimensions
            _, _, Hi, Wi = feature.shape
            
            decoded = decoded.reshape(B, Hi, Wi, -1).permute(0, 3, 1, 2)
            
            # Upsample to the first stage's resolution
            if i > 0:
                _, _, H0, W0 = features[0].shape
                decoded = F.interpolate(decoded, size=(H0, W0), mode='bilinear', align_corners=False)
            
            outs.append(decoded)
        
        # get final output
        out = torch.cat(outs, dim=1)
        out = self.seg_head(out)
        out = self.up(out)  # 4x upsampling to original image size
        
        return out
    
    def predict(self, x, threshold=0.5):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            return (probs > threshold).float()


def test_segformer():
    config = {
        'img_size': 256,
        'in_channels': 3,
        'num_classes': 1,
        'preset': 'tiny',  # Use the tiny preset (large has 25 times more params)
        'dropout': 0.1     
    }
    
    model = SegFormer(config)
    print(f"Model created with preset: {config['preset']}")
    
    x = torch.randn(2, 3, 256, 256)
    
    output = model(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    prediction = model.predict(x)
    print(f"Binary prediction shape: {prediction.shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    return model


if __name__ == "__main__":
    test_segformer()