# training/base_trainer.py
from abc import ABC, abstractmethod
from typing import List
import torch
from training.hooks import TrainingHook

class BaseTrainer(ABC):

    def __init__(self, model, optimizer, loss_fn, hooks: List[TrainingHook] = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.hooks = hooks or []

    def train_epoch(self, dataloader):
        self.model.train()
        for batch in dataloader:
            inputs, labels = batch["inputs"], batch["labels"]
            
            logits, features = self.model(inputs)
            seg_loss = self.loss_fn(logits, labels)

            # --- Hook integration: this is the ONLY change to the training loop ---
            extra_loss = torch.tensor(0.0, device=logits.device)
            for hook in self.hooks:
                contrib = hook.on_batch_end(
                    logits=logits,
                    labels=labels,
                    features=features,
                    modalities=batch.get("modalities"),
                )
                if contrib is not None:
                    extra_loss = extra_loss + contrib

            total_loss = seg_loss + extra_loss
            # ---------------------------------------------------------------------

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

    def _epoch_end_hooks(self, metrics: dict):
        for hook in self.hooks:
            hook.on_epoch_end(metrics)

    @abstractmethod
    def train(self, dataloader, epochs: int): ...