import torch
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
    
class CombinedCrossEntropyDiceLoss(nn.Module):
    def __init__(self, class_weights=None, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = smp.losses.DiceLoss(mode="multiclass")
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets):
        ce = self.ce_loss(preds, targets)
        dice = self.dice_loss(preds, targets)
        return self.ce_weight * ce + self.dice_weight * dice
  
def dice_ce_loss(class_weights=None):
    ce_loss = nn.CrossEntropyLoss(weight=class_weights)
    dice_loss = smp.losses.DiceLoss(mode="multiclass")

    def loss_fn(preds, targets):
        return 0.5 * ce_loss(preds, targets) + 0.5 * dice_loss(preds, targets)

    return loss_fn

# Alternative with focal loss (good for imbalanced classes)
class SMPFocalCombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = smp.losses.DiceLoss(mode='multiclass')
        self.focal_loss = smp.losses.FocalLoss(mode='multiclass')
    
    def forward(self, pred, target):
        return self.dice_loss(pred, target) + self.focal_loss(pred, target)
   
class CDnetv2Loss(nn.Module):
    def __init__(self, loss_fn: nn.Module) -> None:
        super().__init__()
        self.loss_fn = loss_fn

    def forward(self, logits: torch.Tensor, logits_aux,target: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(logits, target)
        loss_aux = self.loss_fn(logits_aux, target)
        total_loss = loss + loss_aux
        return total_loss
