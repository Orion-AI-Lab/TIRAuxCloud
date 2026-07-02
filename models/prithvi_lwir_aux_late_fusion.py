"""
Late-fusion Prithvi model for TIRAuxCloud.

This model separates the thermal image pathway from auxiliary variables:

    thermal LWIR channel
        -> Prithvi-EO-2.0 branch with averaged patch-embedding adaptation
        -> dense Prithvi feature/logit map

    auxiliary channels, e.g. DEM, weather, clear counterpart
        -> small CNN branch
        -> dense auxiliary feature map

    [Prithvi map, auxiliary map]
        -> late fusion head
        -> cloud segmentation logits

Input convention
----------------
The input tensor is expected as:

    x: [B, C, H, W]

where:

    x[:, 0:1, :, :]  = thermal channel
                       e.g. cloudy_Radiance_B10_B11_mean

    x[:, 1:, :, :]   = auxiliary channels
                       e.g. DEM, weather, clear counterpart image, etc.

Therefore, in your config, put the thermal feature first:

    features = [
        "cloudy_Radiance_B10_B11_mean",
        "DEM",
        "weather_temp",
        "clear_Radiance_B10_B11_mean"
    ]

Key difference from early-fusion Prithvi-Aux
--------------------------------------------
Early fusion:
    [LWIR, DEM, weather, clear] -> one modified Prithvi patch embedding

Late fusion here:
    LWIR -> Prithvi
    auxiliaries -> separate CNN
    fusion happens near the output

Dependencies
------------
This file imports PrithviEO2LWIRSegmentation from models/prithvi_eo2_lwir.py.
That file must already exist and support input_mode="adapt_patch_embed_avg".
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.prithvi_eo2_lwir import PrithviEO2LWIRSegmentation
except ImportError:
    # Allows running from inside the models/ directory.
    from prithvi_eo2_lwir import PrithviEO2LWIRSegmentation


class AuxConvBlock(nn.Module):
    """Small CNN block for auxiliary rasters."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_channels is None:
            hidden_channels = out_channels

        layers = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PrithviLWIRAuxLateFusion(nn.Module):
    """Thermal Prithvi branch + auxiliary CNN branch + late fusion head.

    Args:
        num_classes:
            Number of final segmentation classes.
        aux_in_channels:
            Number of auxiliary channels after the first thermal channel.
            For example, if features=[LWIR, DEM, temp, clear], aux_in_channels=3.
        prithvi_backbone:
            TerraTorch Prithvi backbone name, e.g. prithvi_eo_v2_300_tl.
        prithvi_feature_channels:
            Number of dense channels produced by the Prithvi branch before fusion.
            The Prithvi wrapper is used with num_classes=prithvi_feature_channels,
            so its segmentation head acts as a dense feature projection.
        aux_feature_channels:
            Number of dense channels produced by the auxiliary branch before fusion.
        freeze_encoder:
            Whether to freeze the Prithvi encoder/backbone.
        prithvi_decoder:
            Decoder used in the Prithvi TerraTorch branch.
        prithvi_decoder_channels:
            Decoder channels for the Prithvi TerraTorch branch.
        prithvi_img_size:
            Image size expected by the Prithvi backbone configuration.
        aux_dropout:
            Dropout in the auxiliary CNN branch.
        fusion_hidden_channels:
            Hidden channels in the fusion head. If None, use
            prithvi_feature_channels + aux_feature_channels.
    """

    def __init__(
        self,
        num_classes: int = 2,
        aux_in_channels: int = 1,
        prithvi_backbone: str = "prithvi_eo_v2_300_tl",
        prithvi_feature_channels: int = 64,
        aux_feature_channels: int = 32,
        freeze_encoder: bool = False,
        prithvi_decoder: str = "UperNetDecoder",
        prithvi_decoder_channels: int = 256,
        prithvi_img_size: int = 224,
        aux_dropout: float = 0.0,
        fusion_hidden_channels: Optional[int] = None,
    ) -> None:
        super().__init__()

        if aux_in_channels < 1:
            raise ValueError(
                "PrithviLWIRAuxLateFusion expects at least one auxiliary channel. "
                "For thermal-only input, use PrithviEO2LWIRSegmentation directly."
            )

        self.num_classes = num_classes
        self.aux_in_channels = aux_in_channels
        self.prithvi_feature_channels = prithvi_feature_channels
        self.aux_feature_channels = aux_feature_channels
        self.freeze_encoder_requested = freeze_encoder

        # Thermal branch:
        # Only x[:, 0:1] goes here.
        # This keeps the same averaging adaptation:
        # original Prithvi 6-channel patch embedding -> 1-channel patch embedding,
        # initialized by the average of pretrained six HLS channel weights.
        self.prithvi_branch = PrithviEO2LWIRSegmentation(
            num_classes=prithvi_feature_channels,
            backbone=prithvi_backbone,
            input_mode="adapt_patch_embed_avg",
            in_channels=1,
            freeze_encoder=freeze_encoder,
            decoder=prithvi_decoder,
            decoder_channels=prithvi_decoder_channels,
            img_size=prithvi_img_size,
        )

        # Auxiliary branch:
        # x[:, 1:] goes here. This branch is randomly initialized and trained
        # from scratch. It does not use Prithvi pretraining.
        self.aux_branch = AuxConvBlock(
            in_channels=aux_in_channels,
            out_channels=aux_feature_channels,
            hidden_channels=aux_feature_channels,
            dropout=aux_dropout,
        )

        if fusion_hidden_channels is None:
            fusion_hidden_channels = prithvi_feature_channels + aux_feature_channels

        fusion_in_channels = prithvi_feature_channels + aux_feature_channels

        self.fusion_head = nn.Sequential(
            nn.Conv2d(fusion_in_channels, fusion_hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_hidden_channels, num_classes, kernel_size=1),
        )

    @property
    def encoder(self) -> nn.Module:
        """Expose the Prithvi encoder for debugging and compatibility."""
        return self.prithvi_branch.encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B, C, H, W], got {tuple(x.shape)}")

        expected_channels = 1 + self.aux_in_channels
        if x.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} channels: "
                f"1 thermal + {self.aux_in_channels} auxiliary, got {x.shape[1]}"
            )

        thermal = x[:, 0:1, :, :]
        aux = x[:, 1:, :, :]

        prithvi_map = self.prithvi_branch(thermal)
        aux_map = self.aux_branch(aux)

        if prithvi_map.shape[-2:] != aux_map.shape[-2:]:
            prithvi_map = F.interpolate(
                prithvi_map,
                size=aux_map.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        fused = torch.cat([prithvi_map, aux_map], dim=1)
        logits = self.fusion_head(fused)

        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return logits

    def get_param_groups(
        self,
        encoder_lr: float,
        decoder_lr: float,
        weight_decay: float = 0.0,
    ) -> List[Dict]:
        """Return optimizer groups with lower LR for the Prithvi encoder.

        Group 0:
            Prithvi encoder/backbone parameters.

        Group 1:
            Prithvi decoder/projection, auxiliary branch, and fusion head.
        """
        encoder_params = [p for p in self.prithvi_branch.encoder.parameters() if p.requires_grad]
        encoder_param_ids = {id(p) for p in self.prithvi_branch.encoder.parameters()}

        non_encoder_params = [
            p
            for p in self.parameters()
            if p.requires_grad and id(p) not in encoder_param_ids
        ]

        groups: List[Dict] = []

        if encoder_params:
            groups.append(
                {
                    "name": "encoder",
                    "params": encoder_params,
                    "lr": encoder_lr,
                    "weight_decay": weight_decay,
                }
            )

        if non_encoder_params:
            groups.append(
                {
                    "name": "decoder_aux_fusion",
                    "params": non_encoder_params,
                    "lr": decoder_lr,
                    "weight_decay": weight_decay,
                }
            )

        if not groups:
            raise RuntimeError("No trainable parameters found.")

        return groups


if __name__ == "__main__":
    # Smoke test with tiny backbone to reduce memory.
    # Input channels:
    #   channel 0 = LWIR
    #   channels 1..3 = auxiliary variables
    model = PrithviLWIRAuxLateFusion(
        num_classes=2,
        aux_in_channels=3,
        prithvi_backbone="prithvi_eo_v2_tiny_tl",
        prithvi_feature_channels=32,
        aux_feature_channels=16,
        freeze_encoder=True,
        prithvi_img_size=224,
    )

    x = torch.randn(2, 4, 224, 224)
    y = model(x)

    print("input:", tuple(x.shape))
    print("output:", tuple(y.shape))
    print(
        "encoder trainable params:",
        sum(p.numel() for p in model.encoder.parameters() if p.requires_grad),
    )
    print(
        "total trainable params:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )
