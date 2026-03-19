import torch
import numpy as np
import segmentation_models_pytorch as smp
import numpy as np
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
  
def getLossFunction(lossname, class_counts=None, device=None):
    if class_counts is not None:
        #counts is class counts: e.g. np.array([123458639, 10561440, 80741393], dtype=np.float64), 
        #                             np.array([11990540, 1955076, 4404464], dtype=np.float64)
        inv_freq = 1.0 / np.array(class_counts) # Inverse frequency
        weights = inv_freq / inv_freq.mean() # Normalize to make mean = 1 (recommended for CrossEntropyLoss)
        class_weights = torch.tensor(weights, dtype=torch.float32) # Convert to PyTorch tensor
    if lossname=="DiceCECombined":
        return CombinedCrossEntropyDiceLoss(class_weights=class_weights.to(device)).to(device)
    elif lossname=="CrossEntropy":
        return torch.nn.CrossEntropyLoss()
    elif lossname=="CrossEntropyWeights" and class_counts is not None and device is not None:
        return torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    elif lossname=="Focal":
        return SMPFocalCombinedLoss()
    elif lossname=="Dice":
        return smp.losses.DiceLoss(mode='multiclass')
    elif lossname=="CDnetV2Loss":
        return CDnetv2Loss(torch.nn.CrossEntropyLoss())
    else:
        print("Wrong loss keyword or parameters")
        raise

