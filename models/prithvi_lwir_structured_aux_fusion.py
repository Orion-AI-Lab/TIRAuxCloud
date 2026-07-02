"""
Structured late-fusion Prithvi model for TIRAuxCloud.

This model separates inputs according to their physical meaning and effective
resolution:

1. Cloudy LWIR / thermal channel
   -> Prithvi-EO-2.0 branch with averaged patch-embedding adaptation

2. Clear counterpart image
   -> comparison branch using [cloudy, clear, cloudy-clear, abs(cloudy-clear)]

3. DEM
   -> shallow spatial CNN branch

4. Weather
   -> context branch: global pooling + MLP + broadcast to spatial map

Then:
   [Prithvi features, clear-difference features, DEM features, weather context]
   -> fusion head
   -> segmentation logits

Input convention
----------------
The input tensor is expected as:

    x: [B, C, H, W]

The channel order is configurable through constructor indices. By default:

    channel 0 = cloudy LWIR / cloudy_Radiance_B10_B11_mean
    channel 1 = clear counterpart / clear_Radiance_B10_B11_mean
    channel 2 = DEM
    channels 3..C-1 = weather variables

Example feature list:

    features = [
        "cloudy_Radiance_B10_B11_mean",
        "clear_Radiance_B10_B11_mean",
        "DEM",
        "weather_temp",
        "weather_humidity",
        "weather_wind_u",
        "weather_wind_v"
    ]

If you do not have one of the auxiliary groups, pass the relevant index/list as
None or empty:
    clear_idx=None
    dem_idx=None
    weather_indices=[]

Dependencies
------------
Requires models/prithvi_eo2_lwir.py with PrithviEO2Segmentation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from prithvi_eo2 import PrithviEO2Segmentation
   
class ConvBranch(nn.Module):
    """Small spatial CNN branch."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        num_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")

        if hidden_channels is None:
            hidden_channels = out_channels

        layers: List[nn.Module] = []
        ch_in = in_channels

        for block_idx in range(num_blocks):
            ch_out = hidden_channels if block_idx < num_blocks - 1 else out_channels
            layers.extend(
                [
                    nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(ch_out),
                    nn.ReLU(inplace=True),
                ]
            )
            ch_in = ch_out

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WeatherContextBranch(nn.Module):
    """Weather context encoder.

    Weather variables are often much coarser than the 256x256 image grid.
    This branch deliberately removes fake high-frequency interpolation detail
    using global average pooling, then produces a scene-level context vector
    that is broadcast back to HxW.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")

        layers: List[nn.Module] = [
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.extend(
            [
                nn.Linear(hidden_channels, out_channels),
                nn.ReLU(inplace=True),
            ]
        )

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, spatial_size: Sequence[int]) -> torch.Tensor:
        # x: [B, Cw, H, W]
        # pooled: [B, Cw]
        pooled = x.mean(dim=(-2, -1))
        context = self.mlp(pooled)  # [B, out_channels]
        context = context[:, :, None, None]
        return context.expand(-1, -1, spatial_size[0], spatial_size[1])


class PrithviLWIRStructuredAuxFusion(nn.Module):
    """Structured late-fusion model for LWIR + heterogeneous auxiliaries."""

    def __init__(
        self,
        num_classes: int = 2,
        total_in_channels: int = 4,
        thermal_idx: int = 0,
        clear_idx: Optional[int] = 1,
        dem_idx: Optional[int] = 2,
        weather_indices: Optional[Sequence[int]] = None,
        prithvi_backbone: str = "prithvi_eo_v2_300_tl",
        prithvi_feature_channels: int = 64,
        clear_feature_channels: int = 32,
        dem_feature_channels: int = 16,
        weather_feature_channels: int = 16,
        fusion_hidden_channels: Optional[int] = None,
        freeze_encoder: bool = False,
        prithvi_decoder: str = "UperNetDecoder",
        prithvi_decoder_channels: int = 256,
        prithvi_img_size: int = 256,
        branch_dropout: float = 0.0,
        weather_hidden_channels: int = 64,
    ) -> None:
        super().__init__()

        if total_in_channels < 1:
            raise ValueError("total_in_channels must be >= 1")

        if not (0 <= thermal_idx < total_in_channels):
            raise ValueError(
                f"thermal_idx={thermal_idx} out of range for total_in_channels={total_in_channels}"
            )

        self.num_classes = num_classes
        self.total_in_channels = total_in_channels
        self.thermal_idx = thermal_idx
        self.clear_idx = clear_idx
        self.dem_idx = dem_idx
        self.weather_indices = list(weather_indices or [])
        self.freeze_encoder_requested = freeze_encoder

        for name, idx in [("clear_idx", clear_idx), ("dem_idx", dem_idx)]:
            if idx is not None and not (0 <= idx < total_in_channels):
                raise ValueError(
                    f"{name}={idx} out of range for total_in_channels={total_in_channels}"
                )

        for idx in self.weather_indices:
            if not (0 <= idx < total_in_channels):
                raise ValueError(
                    f"weather index {idx} out of range for total_in_channels={total_in_channels}"
                )

        aux_indices = [i for i in [clear_idx, dem_idx] if i is not None] + self.weather_indices
        if len(aux_indices) != len(set(aux_indices)):
            raise ValueError(f"Duplicate auxiliary indices detected: {aux_indices}")

        if thermal_idx in aux_indices:
            raise ValueError(
                "thermal_idx should not also be listed as clear/dem/weather index. "
                "The clear branch internally uses thermal together with clear."
            )

        # Thermal/Prithvi branch. It outputs a dense feature map, not final logits.
        self.prithvi_branch = PrithviEO2Segmentation(
            num_classes=prithvi_feature_channels,
            backbone=prithvi_backbone,
            input_mode="adapt_patch_embed_avg",
            in_channels=1,
            freeze_encoder=freeze_encoder,
            decoder=prithvi_decoder,
            decoder_channels=prithvi_decoder_channels,
            img_size=prithvi_img_size,
        )

        fusion_channels = prithvi_feature_channels

        self.use_clear = clear_idx is not None
        if self.use_clear:
            self.clear_branch = ConvBranch(
                in_channels=4,
                out_channels=clear_feature_channels,
                hidden_channels=clear_feature_channels,
                num_blocks=2,
                dropout=branch_dropout,
            )
            fusion_channels += clear_feature_channels
        else:
            self.clear_branch = None

        self.use_dem = dem_idx is not None
        if self.use_dem:
            self.dem_branch = ConvBranch(
                in_channels=1,
                out_channels=dem_feature_channels,
                hidden_channels=dem_feature_channels,
                num_blocks=2,
                dropout=branch_dropout,
            )
            fusion_channels += dem_feature_channels
        else:
            self.dem_branch = None

        self.use_weather = len(self.weather_indices) > 0
        if self.use_weather:
            self.weather_branch = WeatherContextBranch(
                in_channels=len(self.weather_indices),
                out_channels=weather_feature_channels,
                hidden_channels=weather_hidden_channels,
                dropout=branch_dropout,
            )
            fusion_channels += weather_feature_channels
        else:
            self.weather_branch = None

        if fusion_hidden_channels is None:
            fusion_hidden_channels = fusion_channels

        self.fusion_head = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fusion_hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_hidden_channels, num_classes, kernel_size=1),
        )

    @property
    def encoder(self) -> nn.Module:
        return self.prithvi_branch.encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input [B, C, H, W], got {tuple(x.shape)}")

        if x.shape[1] != self.total_in_channels:
            raise ValueError(
                f"Expected {self.total_in_channels} input channels, got {x.shape[1]}"
            )

        spatial_size = x.shape[-2:]

        thermal = x[:, self.thermal_idx:self.thermal_idx + 1, :, :]
        features: List[torch.Tensor] = []

        prithvi_map = self.prithvi_branch(thermal)
        if prithvi_map.shape[-2:] != spatial_size:
            prithvi_map = F.interpolate(
                prithvi_map,
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            )
        features.append(prithvi_map)

        if self.use_clear:
            clear = x[:, self.clear_idx:self.clear_idx + 1, :, :]
            diff = thermal - clear
            clear_input = torch.cat([thermal, clear, diff, diff.abs()], dim=1)
            clear_map = self.clear_branch(clear_input)
            features.append(clear_map)

        if self.use_dem:
            dem = x[:, self.dem_idx:self.dem_idx + 1, :, :]
            dem_map = self.dem_branch(dem)
            features.append(dem_map)

        if self.use_weather:
            weather = x[:, self.weather_indices, :, :]
            weather_map = self.weather_branch(weather, spatial_size=spatial_size)
            features.append(weather_map)

        fused = torch.cat(features, dim=1)
        logits = self.fusion_head(fused)

        if logits.shape[-2:] != spatial_size:
            logits = F.interpolate(
                logits,
                size=spatial_size,
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
    model = PrithviLWIRStructuredAuxFusion(
        num_classes=2,
        total_in_channels=5,
        thermal_idx=0,
        clear_idx=1,
        dem_idx=2,
        weather_indices=[3, 4],
        prithvi_backbone="prithvi_eo_v2_tiny_tl",
        prithvi_feature_channels=32,
        clear_feature_channels=16,
        dem_feature_channels=8,
        weather_feature_channels=8,
        freeze_encoder=True,
        prithvi_img_size=256,
    )

    x = torch.randn(2, 5, 256, 256)
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
