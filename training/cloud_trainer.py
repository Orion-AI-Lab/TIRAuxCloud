import numpy as np
import torch
from tqdm import tqdm
import random
from libraries.utils import set_seed

from training.base_trainer import BaseTrainer
from training.hooks import TrainingHook
from evaluation.validate import validate_all, record_validation_metrics_to_csv
from model_builder.models_tcloud import save_model_and_log_params
from libraries.utils import get_preds_multi_encoders


class CloudTrainer(BaseTrainer):
    """
    Concrete trainer for cloud segmentation models.

    Overrides train_epoch to handle:
    - (x, y) tuple format from dataloaders (not dict)
    - single and dual-encoder forward passes via get_preds_multi_encoders
    - auxiliary loss outputs (e.g. CDnetV2)

    Implements train() with early stopping, validation, and W&B logging.
    """

    def __init__(self, model, optimizer, loss_fn, params_dict, hooks: list[TrainingHook] = None):
        super().__init__(model, optimizer, loss_fn, hooks)
        self.params_dict = params_dict
        self.device = params_dict["device"]

    def train_epoch(self, dataloader):
        self.model.train()
        train_losses = []

        for x, y in tqdm(dataloader, desc="Training", leave=False):
            y = y.to(self.device)
            preds = get_preds_multi_encoders(self.model, x, self.device)

            # CDnetV2 returns (main_logits, aux_logits)
            if isinstance(preds, tuple):
                seg_loss = self.loss_fn(preds[0], preds[1], y)
                logits = preds[0]
            else:
                seg_loss = self.loss_fn(preds, y)
                logits = preds

            extra_loss = torch.tensor(0.0, device=self.device)
            for hook in self.hooks:
                contrib = hook.on_batch_end(logits=logits, labels=y)
                if contrib is not None:
                    extra_loss = extra_loss + contrib

            total_loss = seg_loss + extra_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            train_losses.append(total_loss.item())

        return float(np.mean(train_losses))

    def train(self, train_loader, val_loader, save_dir=False, wandbrun=None):
        seed = self.params_dict.get("seed", random.randint(0, 2**32 - 1))
        set_seed(seed)
        self.params_dict["seed"] = seed
        patience = self.params_dict["patience"]
        max_epochs = self.params_dict.get("max_epochs", 200)
        target_metric = self.params_dict["target_metric"]

        early_stop_dict = {
            "best_early_stop": 0,
            "epochs_no_improve": 0,
            "patience": patience,
        }
        first_epoch = True

        for epoch in range(max_epochs):
            print(f"\nEpoch {epoch + 1}/{max_epochs}")
            avg_train_loss = self.train_epoch(train_loader)
            print(f"Train loss: {avg_train_loss:.4f}")

            metrics = validate_all(self.model, val_loader, self.params_dict)
            metrics["train_loss"] = avg_train_loss
            metrics["epochs_best"] = epoch

            self._epoch_end_hooks(metrics)

            if first_epoch:
                early_stop_dict["best_early_stop"] = metrics[target_metric] - 1

            early_stop_dict["metrics"] = metrics
            early_stop_dict["early_stop_metric"] = metrics[target_metric]
            early_stop_dict["epoch"] = epoch

            if wandbrun:
                wandbrun.log(metrics)

            early_stop_dict, stop = self._early_stop(early_stop_dict, save_dir, wandbrun)
            first_epoch = False

            if stop:
                break

        best = early_stop_dict["best_early_stop"]
        print(f"Training finished. Best {target_metric}: {best:.4f}")

    def _early_stop(self, early_stop_dict, save_dir, wandbrun):
        target_metric = self.params_dict["target_metric"]

        if early_stop_dict["early_stop_metric"] > early_stop_dict["best_early_stop"]:
            early_stop_dict["best_early_stop"] = early_stop_dict["early_stop_metric"]
            early_stop_dict["epochs_no_improve"] = 0

            if save_dir:
                model_path = save_model_and_log_params(
                    self.model, save_dir, self.params_dict["model_file"]
                )
                self.params_dict["model_file"] = model_path

            best = early_stop_dict["best_early_stop"]
            mess = "saved" if save_dir else ""
            print(f"  New best model {mess} ({target_metric}={best:.4f})")
            early_stop_dict["best_metrics"] = early_stop_dict["metrics"].copy()

        else:
            early_stop_dict["epochs_no_improve"] += 1
            n = early_stop_dict["epochs_no_improve"]

            if n >= early_stop_dict["patience"]:
                print(f"Early stopping triggered after {n} epochs with no improvement.")
                if self.params_dict["results_csv"]:
                    record_validation_metrics_to_csv(
                        self.params_dict["results_csv"],
                        early_stop_dict["best_metrics"],
                        self.params_dict,
                        wandbrun=wandbrun,
                    )
                return early_stop_dict, True

        return early_stop_dict, False
