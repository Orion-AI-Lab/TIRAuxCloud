from abc import ABC, abstractmethod
import torch
from training.hooks import TrainingHook

class BaseTrainer(ABC):

    def __init__(self, model, optimizer, loss_fn, hooks: TrainingHook | None = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.hooks = hooks or []

    def train_epoch(self, dataloader):
        """Run one training epoch. Must be implemented by subclasses."""
        ...

    def _epoch_end_hooks(self, metrics: dict):
        for hook in self.hooks:
            hook.on_epoch_end(metrics)

    @abstractmethod
    def train(self, dataloader, epochs: int): ...