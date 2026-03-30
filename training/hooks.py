from abc import ABC
from typing import Optional
import torch

class TrainingHook(ABC):
    """
    Extension point for training loop features.
    Uncertainty, XAI, and entropy regularization
    all register as hooks — the core loop never changes.
    """

    def on_batch_end(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: Optional[dict] = None,
        modalities: Optional[list] = None,
    ) -> Optional[torch.Tensor]:
        return None

    def on_epoch_end(self, metrics: dict) -> None:
        pass

class UncertaintyHook(TrainingHook):
    def on_epoch_end(self, metrics: dict) -> None:
        uncertainty = metrics.get("mean_uncertainty")
        if uncertainty is not None:
            print(f"[UncertaintyHook] Mean uncertainty: {uncertainty:.4f}")

class EntropyRegHook(TrainingHook):
    """
    Adds multi-scale functional entropy regularization per batch.
    Prevents unimodal dominance (Section IV of the paper).
    Returns extra loss term — summed into total_loss by the trainer.
    """

    def __init__(self, lambda_reg: float = 0.1):
        self.lambda_reg = lambda_reg

    def on_batch_end(self, logits, labels, features=None, modalities=None):
        probs = torch.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        # Maximize entropy = minimize negative entropy
        return -self.lambda_reg * entropy


class AttributionHook(TrainingHook):
    """Placeholder — logs that attribution logging is registered."""

    def on_epoch_end(self, metrics: dict) -> None:
        print("[AttributionHook] Attribution logging registered.")