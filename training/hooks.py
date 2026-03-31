from abc import ABC
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
        features: dict | None = None,
        modalities: list | None = None,
    ) -> torch.Tensor | None:
        return None

    def on_epoch_end(self, metrics: dict) -> None:
        pass

class UncertaintyHook(TrainingHook):
    """
    Tracks mean single-pass entropy uncertainty during training.
    Accumulates per-batch softmax entropy in on_batch_end() — no extra
    forward passes. Reports epoch mean at on_epoch_end().
    Prototype: PR #4. This hook brings the same metric into the training loop.
    """
    def __init__(self):
        self._batch_entropies: list = []

    def on_batch_end(self, logits, labels, features=None, modalities=None):
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            pixel_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            self._batch_entropies.append(pixel_entropy.mean().item())
        return None

    def on_epoch_end(self, metrics: dict) -> None:
        if not self._batch_entropies:
            return
        mean_uncertainty = sum(self._batch_entropies) / len(self._batch_entropies)
        print(f"[UncertaintyHook] Mean uncertainty (epoch): {mean_uncertainty:.4f}")
        metrics["mean_uncertainty"] = mean_uncertainty
        self._batch_entropies.clear()

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