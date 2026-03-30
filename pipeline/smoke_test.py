import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset

from pipeline.runner import build_pipeline

PARAMS = {
    "model_type":    "Unet",
    "features":      ["tir"],
    "num_classes":   2,
    "device":        "cpu",
    "loss":          "CrossEntropy",
    "lr":            1e-3,
    "optimizer":     "adam",
    "patience":      5,
    "target_metric": "iou_avg",
    "results_csv":   None,
    "lambda_reg":    0.1,
}

BATCH_SIZE = 4
H, W = 64, 64

dummy_x = torch.randn(BATCH_SIZE, len(PARAMS["features"]), H, W)
dummy_y = torch.randint(0, PARAMS["num_classes"], (BATCH_SIZE, H, W))
loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=BATCH_SIZE)


def test_forward_backward():
    print("Building pipeline...")
    trainer = build_pipeline(PARAMS)
    print(f"  Model:     {trainer.model.name}")
    print(f"  Optimizer: {type(trainer.optimizer).__name__}")
    print(f"  Loss:      {type(trainer.loss_fn).__name__}")
    print(f"  Hooks:     {[type(h).__name__ for h in trainer.hooks]}")

    print("\nRunning one train_epoch with dummy data...")
    avg_loss = trainer.train_epoch(loader)
    print(f"  avg_loss = {avg_loss:.4f}")

    print("\nSmoke test PASSED — forward + backward + hooks all ran cleanly.")


if __name__ == "__main__":
    test_forward_backward()
