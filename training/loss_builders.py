import torch
import numpy as np
import segmentation_models_pytorch as smp
import numpy as np
import torch
from training.loss_registry import register_loss
from training.loss import CombinedCrossEntropyDiceLoss, SMPFocalCombinedLoss, CDnetv2Loss

def _compute_weights(class_counts, device):
    """Shared helper — only used when class_counts is provided."""
    if class_counts is None:
        return None
    inv_freq = 1.0 / np.array(class_counts)
    weights = inv_freq / inv_freq.mean()
    return torch.tensor(weights, dtype=torch.float32).to(device)

@register_loss("DiceCECombined")
def build_dice_ce(class_counts=None, device=None):
    weights = _compute_weights(class_counts, device)
    return CombinedCrossEntropyDiceLoss(class_weights=weights).to(device)

@register_loss("CrossEntropy")
def build_ce(class_counts=None, device=None):
    return torch.nn.CrossEntropyLoss()

@register_loss("CrossEntropyWeights")
def build_ce_weights(class_counts=None, device=None):
    weights = _compute_weights(class_counts, device)
    return torch.nn.CrossEntropyLoss(weight=weights)

@register_loss("Focal")
def build_focal(class_counts=None, device=None):
    return SMPFocalCombinedLoss()

@register_loss("Dice")
def build_dice(class_counts=None, device=None):
    return smp.losses.DiceLoss(mode='multiclass')

@register_loss("CDnetV2Loss")
def build_cdnetv2(class_counts=None, device=None):
    return CDnetv2Loss(torch.nn.CrossEntropyLoss())
