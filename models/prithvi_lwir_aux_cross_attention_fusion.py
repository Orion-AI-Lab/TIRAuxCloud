
"""
Prithvi LWIR + auxiliary cross-attention fusion model for TIRAuxCloud.

Requires:
    models/prithvi_eo2.py
with:
    class PrithviEO2Segmentation

Main idea:
    cloudy LWIR -> Prithvi branch
    auxiliaries -> compact tokens
    Prithvi dense features query auxiliary tokens through cross-attention

Default input channel order:
    0 = cloudy LWIR / cloudy_Radiance_B10_B11_mean
    1 = clear counterpart / clear_Radiance_B10_B11_mean
    2 = DEM
    3..C-1 = weather variables
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from models.prithvi_eo2 import PrithviEO2Segmentation
except ImportError:
    from prithvi_eo2 import PrithviEO2Segmentation


class SpatialTokenEncoder(nn.Module):
    """Encode a spatial raster/map into compact tokens.

    Input:  [B, C, H, W]
    Output: [B, token_grid * token_grid, embed_dim]
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 64,
        token_grid: int = 4,
        hidden_channels: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if token_grid < 1:
            raise ValueError(f"token_grid must be >= 1, got {token_grid}")

        hidden_channels = hidden_channels or embed_dim

        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, embed_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        ]

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.proj = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((token_grid, token_grid))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.proj(x)                         # [B, d, H, W]
        z = self.pool(z)                         # [B, d, G, G]
        return z.flatten(2).transpose(1, 2)      # [B, G*G, d]


class WeatherTokenEncoder(nn.Module):
    """Encode coarse weather maps as scene-level context tokens.

    This removes fake high-frequency detail from resampled weather maps by
    spatially averaging them first.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 64,
        hidden_channels: int = 64,
        num_tokens: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")
        if num_tokens < 1:
            raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")

        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        layers: List[nn.Module] = [
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.extend(
            [
                nn.Linear(hidden_channels, num_tokens * embed_dim),
                nn.ReLU(inplace=True),
            ]
        )

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(-2, -1))            # [B, Cw]
        tokens = self.mlp(pooled)                # [B, num_tokens*d]
        return tokens.view(x.shape[0], self.num_tokens, self.embed_dim)


class PrithviAuxCrossAttention(nn.Module):
    """Cross-attention from Prithvi spatial features to auxiliary tokens.

    prithvi_map: [B, C_p, H, W]
    aux_tokens:  [B, M, d]
    """

    def __init__(
        self,
        prithvi_channels: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )

        self.q_proj = nn.Conv2d(prithvi_channels, embed_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(embed_dim, prithvi_channels, kernel_size=1)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.q_norm = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()
        self.aux_norm = nn.LayerNorm(embed_dim) if use_layernorm else nn.Identity()

    def forward(self, prithvi_map: torch.Tensor, aux_tokens: torch.Tensor) -> torch.Tensor:
        B, _, H, W = prithvi_map.shape

        q_map = self.q_proj(prithvi_map)          # [B, d, H, W]
        q = q_map.flatten(2).transpose(1, 2)      # [B, H*W, d]

        q = self.q_norm(q)
        aux_tokens = self.aux_norm(aux_tokens)

        attended, _ = self.cross_attn(
            query=q,
            key=aux_tokens,
            value=aux_tokens,
            need_weights=False,
        )

        attended_map = attended.transpose(1, 2).reshape(B, -1, H, W)
        return prithvi_map + self.out_proj(attended_map)


class PrithviLWIRAuxCrossAttentionFusion(nn.Module):
    """Prithvi thermal branch + compact auxiliary-token cross-attention fusion."""

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
        aux_embed_dim: int = 64,
        aux_token_grid: int = 4,
        num_heads: int = 4,
        fusion_hidden_channels: Optional[int] = None,
        freeze_encoder: bool = False,
        prithvi_decoder: str = "UperNetDecoder",
        prithvi_decoder_channels: int = 256,
        prithvi_img_size: int = 256,
        branch_dropout: float = 0.0,
        weather_hidden_channels: int = 64,
        weather_num_tokens: int = 1,
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
                "The clear branch internally uses the thermal channel."
            )

        # Thermal Prithvi branch receives only cloudy LWIR and uses the averaged
        # patch-embedding adaptation from prithvi_eo2.py.
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

        self.use_clear = clear_idx is not None
        self.clear_token_encoder = (
            SpatialTokenEncoder(
                in_channels=4,
                embed_dim=aux_embed_dim,
                token_grid=aux_token_grid,
                dropout=branch_dropout,
            )
            if self.use_clear
            else None
        )

        self.use_dem = dem_idx is not None
        self.dem_token_encoder = (
            SpatialTokenEncoder(
                in_channels=1,
                embed_dim=aux_embed_dim,
                token_grid=aux_token_grid,
                dropout=branch_dropout,
            )
            if self.use_dem
            else None
        )

        self.use_weather = len(self.weather_indices) > 0
        self.weather_token_encoder = (
            WeatherTokenEncoder(
                in_channels=len(self.weather_indices),
                embed_dim=aux_embed_dim,
                hidden_channels=weather_hidden_channels,
                num_tokens=weather_num_tokens,
                dropout=branch_dropout,
            )
            if self.use_weather
            else None
        )

        if not (self.use_clear or self.use_dem or self.use_weather):
            raise ValueError(
                "At least one auxiliary group must be enabled. "
                "Use PrithviEO2Segmentation for thermal-only input."
            )

        self.cross_attention = PrithviAuxCrossAttention(
            prithvi_channels=prithvi_feature_channels,
            embed_dim=aux_embed_dim,
            num_heads=num_heads,
            dropout=branch_dropout,
        )

        fusion_hidden_channels = fusion_hidden_channels or prithvi_feature_channels

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(
                prithvi_feature_channels,
                fusion_hidden_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
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

        prithvi_map = self.prithvi_branch(thermal)
        if prithvi_map.shape[-2:] != spatial_size:
            prithvi_map = F.interpolate(
                prithvi_map,
                size=spatial_size,
                mode="bilinear",
                align_corners=False,
            )

        aux_tokens: List[torch.Tensor] = []

        if self.use_clear:
            clear = x[:, self.clear_idx:self.clear_idx + 1, :, :]
            diff = thermal - clear
            clear_input = torch.cat([thermal, clear, diff, diff.abs()], dim=1)
            aux_tokens.append(self.clear_token_encoder(clear_input))

        if self.use_dem:
            dem = x[:, self.dem_idx:self.dem_idx + 1, :, :]
            aux_tokens.append(self.dem_token_encoder(dem))

        if self.use_weather:
            weather = x[:, self.weather_indices, :, :]
            aux_tokens.append(self.weather_token_encoder(weather))

        aux_tokens_cat = torch.cat(aux_tokens, dim=1)  # [B, M, d]

        prithvi_map = self.cross_attention(
            prithvi_map=prithvi_map,
            aux_tokens=aux_tokens_cat,
        )

        logits = self.segmentation_head(prithvi_map)

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
        """Return optimizer groups with lower LR for Prithvi encoder.

        Group 0:
            Prithvi encoder/backbone parameters.

        Group 1:
            Prithvi decoder/projection, auxiliary token encoders,
            cross-attention, and segmentation head.
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
                    "name": "decoder_aux_cross_attention",
                    "params": non_encoder_params,
                    "lr": decoder_lr,
                    "weight_decay": weight_decay,
                }
            )

        if not groups:
            raise RuntimeError("No trainable parameters found.")

        return groups


if __name__ == "__main__":
    model = PrithviLWIRAuxCrossAttentionFusion(
        num_classes=2,
        total_in_channels=5,
        thermal_idx=0,
        clear_idx=1,
        dem_idx=2,
        weather_indices=[3, 4],
        prithvi_backbone="prithvi_eo_v2_tiny_tl",
        prithvi_feature_channels=32,
        aux_embed_dim=64,
        aux_token_grid=4,
        num_heads=4,
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
