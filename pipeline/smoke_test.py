import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, TensorDataset

from configs.pipeline_config import PipelineConfig
from pipeline.runner import build_pipeline

CONFIG = PipelineConfig(
    model_type="Unet",
    features=["tir"],
    num_classes=2,
    device="cpu",
    loss="CrossEntropy",
    lr=1e-3,
    optimizer="adam",
    patience=5,
    lambda_reg=0.1,
)


def test_forward_backward():
    print("Building pipeline...")

    BATCH_SIZE = 4
    H, W = 64, 64 # Small spatial dims — enough to verify shapes, fast on CPU

    dummy_x = torch.randn(BATCH_SIZE, len(CONFIG.features), H, W)
    dummy_y = torch.randint(0, CONFIG.num_classes, (BATCH_SIZE, H, W))
    loader = DataLoader(TensorDataset(dummy_x, dummy_y), batch_size=BATCH_SIZE)
    trainer = build_pipeline(CONFIG.to_dict())

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
