"""
Prithvi-EO-2.0 integration for TIRAuxCloud.

This module exposes a plain nn.Module:
    model(x) -> segmentation logits [B, num_classes, H, W]

It internally uses TerraTorch's Prithvi-EO-2.0 segmentation factory.

External input expected by your dataloader:
    x: [B, C, H, W]

Supported input modes
---------------------

1) input_mode="adapt_patch_embed_avg"  [DEFAULT]
   - External input: [B, C, H, W]
   - Replaces Prithvi's pretrained 6-channel Conv3d patch embedding with
     a C-channel Conv3d patch embedding.
   - Initializes all C input-channel weights from the average of the
     pretrained six HLS input-channel weights.
   - For LWIR-only, use C=1.
   - For auxiliary inputs, use C=len(featset), e.g. LWIR + DEM + weather.

2) input_mode="native_hls6"
   - External input: [B, 6, H, W]
   - Keeps Prithvi's original six-channel patch embedding unchanged.
   - Use this when you provide the native Prithvi/HLS optical bands:
       BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2
   - For Landsat 8/9 OLI this corresponds to:
       B2, B3, B4, B5, B6, B7
     in exactly that order.

3) input_mode="repeat_to_hls6"
   - External input: [B, 1, H, W]
   - Keeps Prithvi's original six-channel patch embedding unchanged.
   - Repeats the one LWIR channel into all six HLS slots:
       [B,1,H,W] -> [B,6,H,W]
   - Mainly an ablation.

4) input_mode="swir_lwir_zeros"
   - External input: [B, 1, H, W]
   - Keeps Prithvi's original six-channel patch embedding unchanged.
   - Maps the LWIR channel into Prithvi's SWIR_1 and SWIR_2 input slots:
       BLUE       = 0
       GREEN      = 0
       RED        = 0
       NIR_NARROW = 0
       SWIR_1     = LWIR
       SWIR_2     = LWIR

Scientific note
---------------
Prithvi-EO-2.0 was pretrained on six HLS optical bands, not thermal bands:
    BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2

Therefore the LWIR modes are transfer-learning adaptations, not thermal-native
pretraining. The native_hls6 mode is the closest to Prithvi's pretraining
input structure, provided your raw Landsat DN values are converted to
reflectance-like inputs before training.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


PRITHVI_HLS_BANDS = [
    "BLUE",
    "GREEN",
    "RED",
    "NIR_NARROW",
    "SWIR_1",
    "SWIR_2",
]


LANDSAT_OLI_TO_PRITHVI_ORDER = [
    "B2",  # BLUE
    "B3",  # GREEN
    "B4",  # RED
    "B5",  # NIR_NARROW
    "B6",  # SWIR_1
    "B7",  # SWIR_2
]


def _default_neck_indices(backbone: str) -> List[int]:
    """Default intermediate-layer indices used for segmentation necks."""
    name = backbone.lower()

    if "600" in name:
        return [7, 15, 23, 31]

    if "300" in name:
        return [5, 11, 17, 23]

    return [2, 5, 8, 11]


def _extract_pixel_logits(output):
    """Extract segmentation logits from several plausible TerraTorch outputs."""
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in ("out", "output", "logits", "prediction", "predictions"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value

    for attr in ("out", "output", "logits", "prediction", "predictions"):
        value = getattr(output, attr, None)
        if torch.is_tensor(value):
            return value

    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value):
                return value

    raise TypeError(
        "Could not extract segmentation logits from TerraTorch model output. "
        f"Got type={type(output)!r}."
    )


def _get_parent_and_child(root: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    """Return parent module and final child attribute name for a dotted path."""
    parts = qualified_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _find_first_conv3d_with_in_channels(
    model: nn.Module,
    in_channels: int = 6,
) -> Tuple[str, nn.Conv3d]:
    """Find the first Conv3d with a given input-channel count."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv3d) and module.in_channels == in_channels:
            return name, module

    raise RuntimeError(
        f"Could not find Conv3d patch embedding with in_channels={in_channels}. "
        "TerraTorch internals may have changed. Inspect model.named_modules()."
    )


def _replace_first_conv3d_6_to_n_by_average(
    model: nn.Module,
    in_channels: int,
) -> str:
    """Replace first Conv3d(in_channels=6) with Conv3d(in_channels=N).

    The new N-channel weights are initialized from the mean of the six
    pretrained Prithvi HLS input-channel weights.

    Args:
        model:
            TerraTorch pixel model.
        in_channels:
            Number of external input channels, e.g. 1 for LWIR-only,
            2 for LWIR+DEM, 5 for LWIR+DEM+weather+clear counterpart.

    Returns:
        Qualified module name that was replaced.
    """
    if in_channels < 1:
        raise ValueError(f"in_channels must be >= 1, got {in_channels}")

    target_name, old = _find_first_conv3d_with_in_channels(model, in_channels=6)

    new = nn.Conv3d(
        in_channels=in_channels,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=old.bias is not None,
        padding_mode=old.padding_mode,
    )

    with torch.no_grad():
        # old.weight shape: [out_channels, 6, t, h, w]
        # mean_weight shape: [out_channels, 1, t, h, w]
        mean_weight = old.weight.mean(dim=1, keepdim=True)

        # new.weight shape: [out_channels, in_channels, t, h, w]
        new.weight.copy_(mean_weight.repeat(1, in_channels, 1, 1, 1))

        if old.bias is not None:
            new.bias.copy_(old.bias)

    parent, child_name = _get_parent_and_child(model, target_name)
    setattr(parent, child_name, new)

    return target_name


def _expand_lwir_to_swir_slots(
    x: torch.Tensor,
    total_channels: int = 6,
    swir1_idx: int = 4,
    swir2_idx: int = 5,
) -> torch.Tensor:
    """Map one LWIR channel into Prithvi's SWIR_1 and SWIR_2 slots.

    Prithvi HLS band order:
        0 BLUE
        1 GREEN
        2 RED
        3 NIR_NARROW
        4 SWIR_1
        5 SWIR_2

    Input:
        x: [B, 1, H, W]

    Output:
        out: [B, 6, H, W]
             BLUE/GREEN/RED/NIR_NARROW = 0
             SWIR_1 = x
             SWIR_2 = x
    """
    if x.ndim != 4:
        raise ValueError(f"Expected input tensor [B, 1, H, W], got {tuple(x.shape)}")

    if x.shape[1] != 1:
        raise ValueError(
            f"swir_lwir_zeros mode expects exactly 1 input channel, got {x.shape[1]}"
        )

    if not (0 <= swir1_idx < total_channels and 0 <= swir2_idx < total_channels):
        raise ValueError(
            f"Invalid SWIR indices swir1_idx={swir1_idx}, swir2_idx={swir2_idx}, "
            f"total_channels={total_channels}"
        )

    out = torch.zeros(
        x.shape[0],
        total_channels,
        x.shape[2],
        x.shape[3],
        device=x.device,
        dtype=x.dtype,
    )

    out[:, swir1_idx:swir1_idx + 1, :, :] = x
    out[:, swir2_idx:swir2_idx + 1, :, :] = x

    return out


def _find_backbone_module(pixel_model: nn.Module) -> Tuple[nn.Module, str]:
    """Try to locate TerraTorch's backbone module for freezing/param groups."""
    for name in ("backbone", "encoder"):
        module = getattr(pixel_model, name, None)
        if isinstance(module, nn.Module):
            return module, name

    for name, module in pixel_model.named_children():
        if name in ("backbone", "encoder") and isinstance(module, nn.Module):
            return module, name

    raise RuntimeError(
        "Could not locate a 'backbone' or 'encoder' module in the TerraTorch "
        "pixel-wise model. Print pixel_model to inspect the current TerraTorch layout."
    )


class PrithviEO2Segmentation(nn.Module):
    """Prithvi-EO-2.0 segmentation wrapper for native HLS6, LWIR, and auxiliaries."""

    VALID_INPUT_MODES = {
        "adapt_patch_embed_avg",
        "native_hls6",
        "repeat_to_hls6",
        "swir_lwir_zeros",
    }

    def __init__(
        self,
        num_classes: int = 2,
        backbone: str = "prithvi_eo_v2_300_tl",
        input_mode: str = "adapt_patch_embed_avg",
        in_channels: int = 1,
        freeze_encoder: bool = False,
        decoder: str = "UperNetDecoder",
        decoder_channels: int = 256,
        decoder_scale_modules: bool = True,
        head_dropout: float = 0.1,
        img_size: int = 256,
        neck_indices: Optional[Sequence[int]] = None,
        rescale: bool = True,
        pretrained: bool = True,
        coords_encoding: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()

        if input_mode not in self.VALID_INPUT_MODES:
            raise ValueError(
                f"Unsupported input_mode={input_mode!r}. "
                f"Valid values: {sorted(self.VALID_INPUT_MODES)}"
            )

        try:
            from terratorch.tasks import SemanticSegmentationTask
        except Exception as exc:
            raise ImportError(
                f"Could not import TerraTorch or one of its dependencies: {exc}"
            ) from exc

        self.num_classes = num_classes
        self.backbone_name = backbone
        self.input_mode = input_mode
        self.in_channels = in_channels
        self.freeze_encoder_requested = freeze_encoder

        if neck_indices is None:
            neck_indices = _default_neck_indices(backbone)

        if coords_encoding is None:
            # Your current TIRAuxCloud-style pipeline does not pass date/location
            # metadata, so keep metadata encoding disabled by default.
            coords_encoding = []

        model_args = {
            "backbone": backbone,
            "backbone_pretrained": pretrained,
            "backbone_img_size": img_size,
            "backbone_num_frames": 1,
            "backbone_coords_encoding": list(coords_encoding),
            "backbone_bands": PRITHVI_HLS_BANDS,
            "decoder": decoder,
            "decoder_channels": decoder_channels,
            "decoder_scale_modules": decoder_scale_modules,
            "num_classes": num_classes,
            "rescale": rescale,
            "head_dropout": head_dropout,
            "necks": [
                {"name": "SelectIndices", "indices": list(neck_indices)},
                {"name": "ReshapeTokensToImage"},
            ],
        }

        # We use the Task only as TerraTorch's model factory. Your existing
        # training loop remains responsible for loss/optimizer/scheduler.
        task = SemanticSegmentationTask(
            model_factory="EncoderDecoderFactory",
            model_args=model_args,
            loss="ce",
            freeze_backbone=False,
            freeze_decoder=False,
        )

        self.pixel_model = task.model

        self.replaced_patch_embed_name: Optional[str] = None

        if input_mode == "adapt_patch_embed_avg":
            # Flexible/default mode:
            # input [B, N, H, W]
            # Prithvi patch embedding 6 channels -> N channels.
            # This includes your tested one-channel LWIR averaging case.
            self.replaced_patch_embed_name = _replace_first_conv3d_6_to_n_by_average(
                self.pixel_model,
                in_channels=in_channels,
            )

        elif input_mode == "native_hls6":
            if in_channels != 6:
                raise ValueError(
                    f"native_hls6 expects in_channels=6 "
                    f"with Landsat B2,B3,B4,B5,B6,B7 / HLS bands, got {in_channels}"
                )
            # Keep original Prithvi 6-channel patch embedding unchanged.

        elif input_mode == "repeat_to_hls6":
            if in_channels != 1:
                raise ValueError(
                    f"repeat_to_hls6 expects external in_channels=1, got {in_channels}"
                )
            # Keep original Prithvi 6-channel patch embedding unchanged.

        elif input_mode == "swir_lwir_zeros":
            if in_channels != 1:
                raise ValueError(
                    f"swir_lwir_zeros expects external in_channels=1, got {in_channels}"
                )
            # Keep original Prithvi 6-channel patch embedding unchanged.

        self.encoder, self.encoder_attr_name = _find_backbone_module(self.pixel_model)

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f"Expected input tensor [B, C, H, W], got shape={tuple(x.shape)}"
            )

        if self.input_mode == "adapt_patch_embed_avg":
            if x.shape[1] != self.in_channels:
                raise ValueError(
                    f"adapt_patch_embed_avg expected {self.in_channels} channels, "
                    f"got {x.shape[1]}"
                )
            return x

        if self.input_mode == "native_hls6":
            if x.shape[1] != 6:
                raise ValueError(
                    "native_hls6 expected 6 channels in order "
                    "[B2, B3, B4, B5, B6, B7] / "
                    "[BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2], "
                    f"got {x.shape[1]}"
                )
            return x

        if self.input_mode == "repeat_to_hls6":
            if x.shape[1] != 1:
                raise ValueError(
                    f"repeat_to_hls6 expected 1 channel, got {x.shape[1]}"
                )
            return x.repeat(1, 6, 1, 1)

        if self.input_mode == "swir_lwir_zeros":
            return _expand_lwir_to_swir_slots(x)

        raise RuntimeError(f"Unexpected input_mode={self.input_mode!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(x)
        output = self.pixel_model(x)
        return _extract_pixel_logits(output)

    def get_param_groups(
        self,
        encoder_lr: float,
        decoder_lr: float,
        weight_decay: float = 0.0,
    ) -> List[Dict]:
        """Return optimizer groups: encoder vs non-encoder parameters."""
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
        encoder_param_ids = {id(p) for p in self.encoder.parameters()}

        decoder_params = [
            p
            for p in self.pixel_model.parameters()
            if p.requires_grad and id(p) not in encoder_param_ids
        ]

        groups = []

        if encoder_params:
            groups.append(
                {
                    "name": "encoder",
                    "params": encoder_params,
                    "lr": encoder_lr,
                    "weight_decay": weight_decay,
                }
            )

        if decoder_params:
            groups.append(
                {
                    "name": "decoder",
                    "params": decoder_params,
                    "lr": decoder_lr,
                    "weight_decay": weight_decay,
                }
            )

        if not groups:
            raise RuntimeError("No trainable parameters found for optimizer groups.")

        return groups


if __name__ == "__main__":
    # Use tiny backbone for quick smoke tests. Switch to prithvi_eo_v2_300_tl
    # for actual experiments.
    x1 = torch.randn(2, 1, 224, 224)

    model_default = PrithviEO2Segmentation(
        num_classes=2,
        backbone="prithvi_eo_v2_tiny_tl",
        input_mode="adapt_patch_embed_avg",
        in_channels=1,
        freeze_encoder=True,
        img_size=224,
    )
    y1 = model_default(x1)
    print("adapt_patch_embed_avg:", tuple(x1.shape), "->", tuple(y1.shape))
    print("patch embed replaced:", model_default.replaced_patch_embed_name)
    print(
        "encoder trainable:",
        sum(p.numel() for p in model_default.encoder.parameters() if p.requires_grad),
    )

    model_swir = PrithviEO2Segmentation(
        num_classes=2,
        backbone="prithvi_eo_v2_tiny_tl",
        input_mode="swir_lwir_zeros",
        in_channels=1,
        freeze_encoder=True,
        img_size=224,
    )
    y2 = model_swir(x1)
    print("swir_lwir_zeros:", tuple(x1.shape), "->", tuple(y2.shape))

    x_aux = torch.randn(2, 4, 224, 224)
    model_aux = PrithviEO2Segmentation(
        num_classes=2,
        backbone="prithvi_eo_v2_tiny_tl",
        input_mode="adapt_patch_embed_avg",
        in_channels=4,
        freeze_encoder=True,
        img_size=224,
    )
    y3 = model_aux(x_aux)
    print("aux adapt_patch_embed_avg:", tuple(x_aux.shape), "->", tuple(y3.shape))

    x_hls6 = torch.randn(2, 6, 224, 224)
    model_hls6 = PrithviEO2Segmentation(
        num_classes=2,
        backbone="prithvi_eo_v2_tiny_tl",
        input_mode="native_hls6",
        in_channels=6,
        freeze_encoder=True,
        img_size=224,
    )
    y4 = model_hls6(x_hls6)
    print("native_hls6:", tuple(x_hls6.shape), "->", tuple(y4.shape))
