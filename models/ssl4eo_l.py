"""
SSL4EO-L backbone integration for TIRAuxCloud.

This module defines a segmentation model that uses a TorchGeo SSL4EO-L
pretrained ResNet encoder and a lightweight U-Net style decoder.

Supported input modes
---------------------
1) Default: "adapt_conv_avg"
   - External input: [B, 1, H, W]
   - Replaces the pretrained multi-band conv1 with a 1-channel conv1
   - Initializes that 1-channel conv1 by averaging the pretrained conv1
     weights across input bands
   - This preserves the behaviour already used/tested in the existing pipeline

2) "zero_pad_b10_b11_mean"
   - External input: [B, 1, H, W]
   - Interprets the single input channel as averaged B10-B11 radiance
   - Keeps SSL4EO-L's original OLI/TIRS 11-channel conv1 unchanged
   - Expands input to [B, 11, H, W]:
        B1-B9  = 0
        B10    = input
        B11    = input
   - This is useful when you want to preserve the original pretrained
     band-specific input stem as much as possible.

Expected model output:
    logits: Tensor[B, num_classes, H, W]
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _get_torchgeo_resnet_weights(backbone: str, weights_name: str):
    """Resolve TorchGeo weights matching the selected backbone."""
    try:
        from torchgeo.models import ResNet18_Weights, ResNet50_Weights
    except Exception as exc:
        raise ImportError(
            "Could not import TorchGeo ResNet weights. Install TorchGeo first."
        ) from exc

    if backbone == "resnet18":
        enum_cls = ResNet18_Weights
    elif backbone == "resnet50":
        enum_cls = ResNet50_Weights
    else:
        raise ValueError(
            f"Unsupported backbone={backbone!r}. "
            "Supported values: 'resnet18', 'resnet50'."
        )

    if hasattr(enum_cls, weights_name):
        return getattr(enum_cls, weights_name)

    available = [w.name for w in enum_cls]
    raise ValueError(
        f"Unknown weights_name={weights_name!r} for backbone={backbone!r}. "
        f"Available weights for {backbone}: {available}"
    )


def _build_torchgeo_resnet(backbone: str, weights_name: str) -> nn.Module:
    """Build TorchGeo ResNet encoder with matching SSL4EO-L weights."""
    try:
        from torchgeo.models import resnet18, resnet50
    except Exception as exc:
        raise ImportError(
            "Could not import TorchGeo ResNet models. Install TorchGeo first."
        ) from exc

    weights = _get_torchgeo_resnet_weights(
        backbone=backbone,
        weights_name=weights_name,
    )

    if backbone == "resnet18":
        return resnet18(weights=weights)

    if backbone == "resnet50":
        return resnet50(weights=weights)

    raise ValueError(
        f"Unsupported backbone={backbone!r}. "
        "Supported values: 'resnet18', 'resnet50'."
    )


def _convert_first_conv_to_n_channels(model: nn.Module, in_channels: int = 1) -> None:
    """Convert ResNet conv1 to a different number of input channels.

    For in_channels=1, initialize by averaging pretrained channel weights.
    For in_channels>1, initialize by repeating the mean pretrained channel.
    """
    if not hasattr(model, "conv1"):
        raise AttributeError("Expected ResNet-like model with attribute 'conv1'.")

    old_conv = model.conv1
    if not isinstance(old_conv, nn.Conv2d):
        raise TypeError("Expected model.conv1 to be nn.Conv2d.")

    if old_conv.in_channels == in_channels:
        return

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        padding_mode=old_conv.padding_mode,
    )

    with torch.no_grad():
        mean_weight = old_conv.weight.mean(dim=1, keepdim=True)

        if in_channels == 1:
            new_conv.weight.copy_(mean_weight)
        else:
            new_conv.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1))

        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)

    model.conv1 = new_conv


def _expand_b10_b11_mean_to_oli_tirs(
    x: torch.Tensor,
    total_channels: int = 11,
    b10_idx: int = 9,
    b11_idx: int = 10,
) -> torch.Tensor:
    """Expand averaged B10-B11 radiance to SSL4EO-L OLI/TIRS layout."""
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor [B, 1, H, W], got shape={tuple(x.shape)}")

    if x.shape[1] != 1:
        raise ValueError(
            "zero_pad_b10_b11_mean mode expects exactly one input channel "
            f"(averaged B10-B11 radiance), got {x.shape[1]} channels."
        )

    if not (0 <= b10_idx < total_channels and 0 <= b11_idx < total_channels):
        raise ValueError(
            f"Invalid thermal indices b10_idx={b10_idx}, b11_idx={b11_idx}, "
            f"total_channels={total_channels}."
        )

    out = torch.zeros(
        x.shape[0],
        total_channels,
        x.shape[2],
        x.shape[3],
        device=x.device,
        dtype=x.dtype,
    )

    out[:, b10_idx:b10_idx + 1, :, :] = x
    out[:, b11_idx:b11_idx + 1, :, :] = x

    return out


class ConvBlock(nn.Module):
    """Simple 2-layer convolution block for the decoder."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    """Upsample, concatenate skip connection, then convolve."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SSL4EOLResNetUNet(nn.Module):
    """SSL4EO-L ResNet encoder + U-Net decoder."""

    VALID_INPUT_MODES = {
        "adapt_conv_avg",
        "zero_pad_b10_b11_mean",
    }

    def __init__(
        self,
        num_classes: int = 1,
        backbone: str = "resnet18",
        weights_name: str = "LANDSAT_OLI_TIRS_TOA_MOCO",
        in_channels: int = 1,
        freeze_encoder: bool = False,
        input_mode: str = "adapt_conv_avg",
        thermal_indices: Tuple[int, int] = (9, 10),
    ) -> None:
        super().__init__()

        if input_mode not in self.VALID_INPUT_MODES:
            raise ValueError(
                f"Unsupported input_mode={input_mode!r}. "
                f"Valid values: {sorted(self.VALID_INPUT_MODES)}"
            )

        self.backbone_name = backbone
        self.weights_name = weights_name
        self.in_channels = in_channels
        self.input_mode = input_mode
        self.thermal_indices = thermal_indices

        encoder = _build_torchgeo_resnet(
            backbone=backbone,
            weights_name=weights_name,
        )

        if input_mode == "adapt_conv_avg":
            _convert_first_conv_to_n_channels(encoder, in_channels=in_channels)

        elif input_mode == "zero_pad_b10_b11_mean":
            if in_channels != 1:
                raise ValueError(
                    "zero_pad_b10_b11_mean mode expects external in_channels=1, "
                    f"got in_channels={in_channels}."
                )

            expected_in = encoder.conv1.in_channels
            if expected_in != 11:
                raise ValueError(
                    "zero_pad_b10_b11_mean mode was designed for "
                    "LANDSAT_OLI_TIRS_TOA_MOCO-style 11-channel input. "
                    f"The loaded encoder conv1 expects {expected_in} channels."
                )

        self.encoder = encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        if backbone == "resnet18":
            channels = [64, 64, 128, 256, 512]
            decoder_channels = [256, 128, 64, 64]
        elif backbone == "resnet50":
            channels = [64, 256, 512, 1024, 2048]
            decoder_channels = [512, 256, 128, 64]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.up4 = UpBlock(channels[4], channels[3], decoder_channels[0])
        self.up3 = UpBlock(decoder_channels[0], channels[2], decoder_channels[1])
        self.up2 = UpBlock(decoder_channels[1], channels[1], decoder_channels[2])
        self.up1 = UpBlock(decoder_channels[2], channels[0], decoder_channels[3])

        self.segmentation_head = nn.Conv2d(decoder_channels[3], num_classes, kernel_size=1)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare external input for the encoder input stem."""
        if self.input_mode == "adapt_conv_avg":
            return x

        if self.input_mode == "zero_pad_b10_b11_mean":
            b10_idx, b11_idx = self.thermal_indices
            return _expand_b10_b11_mean_to_oli_tirs(
                x,
                total_channels=self.encoder.conv1.in_channels,
                b10_idx=b10_idx,
                b11_idx=b11_idx,
            )

        raise RuntimeError(f"Unexpected input_mode={self.input_mode!r}")

    def _encode(self, x: torch.Tensor):
        """Return ResNet feature pyramid."""
        x0 = self.encoder.conv1(x)
        x0 = self.encoder.bn1(x0)

        if hasattr(self.encoder, "relu"):
            x0 = self.encoder.relu(x0)
        elif hasattr(self.encoder, "act1"):
            x0 = self.encoder.act1(x0)
        else:
            x0 = F.relu(x0, inplace=True)

        x1 = self.encoder.maxpool(x0)
        x1 = self.encoder.layer1(x1)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)

        return x0, x1, x2, x3, x4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]

        x = self._prepare_input(x)
        x0, x1, x2, x3, x4 = self._encode(x)

        d4 = self.up4(x4, x3)
        d3 = self.up3(d4, x2)
        d2 = self.up2(d3, x1)
        d1 = self.up1(d2, x0)

        logits = self.segmentation_head(d1)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)

        return logits

    def get_param_groups(
        self,
        encoder_lr: float,
        decoder_lr: float,
        weight_decay: float = 0.0,
    ) -> List[Dict]:
        """Return optimizer parameter groups with lower LR for pretrained encoder."""
        decoder_params: Iterable[nn.Parameter] = (
            list(self.up4.parameters())
            + list(self.up3.parameters())
            + list(self.up2.parameters())
            + list(self.up1.parameters())
            + list(self.segmentation_head.parameters())
        )

        return [
            {
                "params": [p for p in self.encoder.parameters() if p.requires_grad],
                "lr": encoder_lr,
                "weight_decay": weight_decay,
            },
            {
                "params": [p for p in decoder_params if p.requires_grad],
                "lr": decoder_lr,
                "weight_decay": weight_decay,
            },
        ]


if __name__ == "__main__":
    model_default = SSL4EOLResNetUNet(
        num_classes=2,
        backbone="resnet18",
        weights_name="LANDSAT_OLI_TIRS_TOA_MOCO",
        in_channels=1,
        input_mode="adapt_conv_avg",
    )
    x = torch.randn(2, 1, 224, 224)
    y = model_default(x)
    print("adapt_conv_avg:", x.shape, "->", y.shape)
    print("adapt_conv_avg conv1:", tuple(model_default.encoder.conv1.weight.shape))

    model_zero_pad = SSL4EOLResNetUNet(
        num_classes=2,
        backbone="resnet18",
        weights_name="LANDSAT_OLI_TIRS_TOA_MOCO",
        in_channels=1,
        input_mode="zero_pad_b10_b11_mean",
    )
    y2 = model_zero_pad(x)
    print("zero_pad_b10_b11_mean:", x.shape, "->", y2.shape)
    print("zero_pad_b10_b11_mean conv1:", tuple(model_zero_pad.encoder.conv1.weight.shape))
