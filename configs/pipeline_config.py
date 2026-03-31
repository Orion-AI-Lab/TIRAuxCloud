from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

VALID_LOSSES = {
    "CrossEntropy", "CrossEntropyWeights",
    "DiceCECombined", "Dice", "Focal", "CDnetV2Loss",
}
VALID_OPTIMIZERS = {"adam", "adamw"}


@dataclass
class PipelineConfig:

    model_type: str = "Unet"
    features: list[str] = field(default_factory=lambda: ["tir"])
    num_classes: int = 2
    traintest: str = "train"

    dataset: str = ""
    dataset_folder: str = ""
    dataset_dir: str | None = None
    target_band: str = "cloud_mask"
    batch_size: int = 64
    cpuworkers: int = 4
    thin_cloud_class: int = -1
    yshift: int = 0
    transform: str | None = None

    loss: str = "CrossEntropy"
    class_counts: list[int] | None = None

    optimizer: str = "adam"
    lr: float = 1e-4
    weight_decay: float = 0.0

    lambda_reg: float = 0.1

    patience: int = 20
    max_epochs: int = 200
    target_metric: str = "iou_avg"
    seed: int | None = None

    device: str = "cpu"
    results_csv: str | None = None
    model_file: str | None = None
    save_model: bool = False

    def __post_init__(self):
        """Validate fields immediately after construction."""
        if not self.features:
            raise ValueError("features must be a non-empty list of band names.")
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}.")
        if self.loss not in VALID_LOSSES:
            raise ValueError(f"Unknown loss '{self.loss}'. Valid: {sorted(VALID_LOSSES)}")
        if self.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(f"Unknown optimizer '{self.optimizer}'. Valid: {sorted(VALID_OPTIMIZERS)}")
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}.")
        if self.patience <= 0:
            raise ValueError(f"patience must be > 0, got {self.patience}.")
        if self.max_epochs <= 0:
            raise ValueError(f"max_epochs must be > 0, got {self.max_epochs}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {self.batch_size}.")
        if not 0.0 <= self.lambda_reg <= 1.0:
            raise ValueError(f"lambda_reg must be in [0, 1], got {self.lambda_reg}.")
        if self.traintest not in {"train", "val", "test"}:
            raise ValueError(
                f"traintest must be 'train', 'val', or 'test', got '{self.traintest}'"
            )

    @classmethod
    def from_json(cls, path: str, key: str = None) -> PipelineConfig:
        """
        Load a PipelineConfig from a JSON file.

        Args:
            path: Path to the JSON file.
            key:  Top-level key to read from (e.g. "viirs_unet").
                  If None, the file must be a flat dict of fields.
        """
        with open(path) as f:
            data = json.load(f)

        if key is not None:
            if key not in data:
                raise KeyError(f"Key '{key}' not found in {path}. Available: {list(data.keys())}")
            data = data[key]

        known = cls.__dataclass_fields__.keys()
        unknown = set(data.keys()) - set(known)
        if unknown:
            raise ValueError(f"Unknown fields in config: {unknown}. Check for typos.")

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "model_type":       self.model_type,
            "features":         self.features,
            "num_classes":      self.num_classes,
            "dataset":          self.dataset,
            "dataset_folder":   self.dataset_folder,
            "target_band":      self.target_band,
            "batch_size":       self.batch_size,
            "cpuworkers":       self.cpuworkers,
            "thin_cloud_class": self.thin_cloud_class,
            "yshift":           self.yshift,
            "transform":        self.transform,
            "loss":             self.loss,
            "optimizer":        self.optimizer,
            "lr":               self.lr,
            "weight_decay":     self.weight_decay,
            "lambda_reg":       self.lambda_reg,
            "patience":         self.patience,
            "max_epochs":       self.max_epochs,
            "target_metric":    self.target_metric,
            "device":           self.device,
            "results_csv":      self.results_csv,
            "traintest":        self.traintest,
            "save_model": self.save_model,
        }
        if self.dataset_dir is not None:
            d["dataset_dir"] = self.dataset_dir
        if self.class_counts is not None:
            d["class_counts"] = self.class_counts
        if self.seed is not None:
            d["seed"] = self.seed
        if self.model_file is not None:
            d["model_file"] = self.model_file
        return d
