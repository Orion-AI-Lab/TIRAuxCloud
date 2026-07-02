import kornia.losses as kl
import torch.nn as nn
import segmentation_models_pytorch as smp

class DiceBoundaryLoss(nn.Module):
    """
    Dice + boundary-aware Hausdorff loss.

    Good for cloud segmentation where region overlap matters,
    but cloud edges should receive extra penalty.
    """

    def __init__(
        self,
        dice_weight=1.0,
        border_weight=0.3,
        alpha=2.0,
        k=10,
    ):
        super().__init__()

        self.dice = smp.losses.DiceLoss(mode="multiclass")
        self.boundary = kl.HausdorffERLoss(
            alpha=alpha,
            k=k,
            reduction="mean",
        )

        self.dice_weight = dice_weight
        self.border_weight = border_weight

    def forward(self, y_pred, y_true):
        """
        y_pred: [B, C, H, W] logits
        y_true: [B, H, W] class ids
        """

        y_true = y_true.long()

        dice_loss = self.dice(y_pred, y_true)

        # Kornia HausdorffERLoss wants target as [B, 1, H, W],
        # not [B, H, W] and not one-hot.
        if y_true.ndim == 3:
            y_true_boundary = y_true.unsqueeze(1)
        elif y_true.ndim == 4 and y_true.shape[1] == 1:
            y_true_boundary = y_true
        else:
            raise ValueError(f"Unexpected target shape for boundary loss: {y_true.shape}")

        boundary_loss = self.boundary(y_pred, y_true_boundary)

        return self.dice_weight * dice_loss + self.border_weight * boundary_loss