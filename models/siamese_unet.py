import torch
import torch.nn as nn
from model_builder.base_model import BaseModel
import segmentation_models_pytorch as smp

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        super().__init__()
        # Note: in_channels here is the TOTAL after concatenation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x):
        # x is already upsampled and concatenated before being passed here
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x

class UnetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, n_blocks=5, use_batchnorm=True):
        super().__init__()
        
        encoder_channels = encoder_channels[::-1]  # [1024, 512, 256, 128, 128, 4]
        
        self.blocks = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # First block: bottleneck (no skip)
        self.blocks.append(DecoderBlock(encoder_channels[0], decoder_channels[0], use_batchnorm))
        
        # Remaining blocks with skip connections
        for i in range(1, n_blocks):
            # Upsampling layer
            self.ups.append(nn.ConvTranspose2d(decoder_channels[i-1], decoder_channels[i-1], kernel_size=2, stride=2))
            
            # Calculate total input channels after upsampling + skip concatenation
            upsampled_channels = decoder_channels[i-1]
            skip_channels = encoder_channels[i] if i < len(encoder_channels) else 0
            total_channels = upsampled_channels + skip_channels
            
            self.blocks.append(DecoderBlock(total_channels, decoder_channels[i], use_batchnorm))
    
    def forward(self, *features):
        features = features[::-1]  # Reverse: deepest first
        
        # First block (bottleneck)
        x = self.blocks[0](features[0])
        
        # Remaining blocks with skip connections
        for i in range(1, len(self.blocks)):
            # Upsample
            x = self.ups[i-1](x)
            
            # Add skip connection if available
            if i < len(features):
                x = torch.cat([x, features[i]], dim=1)
            
            # Process through decoder block
            x = self.blocks[i](x)
        
        return x

class SiameseUNet(BaseModel, nn.Module):
    def __init__(
        self,
        encoder_name="resnet34",
        in_channels=1,
        num_classes=3,
        #encoder_weights="imagenet",
        encoder_weights=None,
        decoder_channels=[256, 128, 64, 32, 16, 8],
        activation=None,
    ):
        super().__init__()

        # Create two separate encoders instead of one shared encoder
        self.encoder_cloudy = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=len(decoder_channels)-1,
            weights=encoder_weights,
        )
        
        self.encoder_clear = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=len(decoder_channels)-1,
            weights=encoder_weights,
        )

        # Get encoder channels from one encoder (both should have same architecture)
        encoder_channels = self.encoder_cloudy.out_channels
        merged_channels = [c * 2 for c in encoder_channels]

        self.decoder = UnetDecoder(
            encoder_channels=merged_channels,
            decoder_channels=decoder_channels,
            n_blocks=len(decoder_channels),
            use_batchnorm=True,
        )

        self.segmentation_head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)
        self.activation = smp.base.SegmentationHead(activation=activation) if activation else None

    def forward(self, cloudy, clear):
        # Use separate encoders for each input
        feats_cloudy = self.encoder_cloudy(cloudy)
        feats_clear = self.encoder_clear(clear)

        #print(f"Encoder out_channels: {self.encoder_cloudy.out_channels}")
        #print(f"Actual encoder outputs:")
        #for i, (f1, f2) in enumerate(zip(feats_cloudy, feats_clear)):
        #    print(f"  Level {i}: {f1.shape}")

        merged_feats = [torch.cat([f1, f2], dim=1) for f1, f2 in zip(feats_cloudy, feats_clear)]

        #print(f"Merged features shapes: {[f.shape for f in merged_feats]}")

        x = self.decoder(*merged_feats)
        logits = self.segmentation_head(x)

        if self.activation is not None:
            return self.activation(logits)
        return logits
    
    @property
    def name(self) -> str:
        return "Siamese"
    
    @classmethod
    def from_config(cls, config):    
        return cls(
            encoder_name="resnet34",
            in_channels=1, 
            num_classes=config["num_classes"],
            encoder_weights=None, 
            activation=None
        )
