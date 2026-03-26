from abc import ABC, abstractmethod
from typing import Dict, Any

import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Contract that every TIRAuxCloud model must satisfy.

    Subclass this and implement the three abstract members.
    Python will raise TypeError at import time if any are missing.
    """

    @abstractmethod
    def forward(self, *inputs):
        """Define the forward pass."""
        ...

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict[str, Any]) -> "BaseModel":
        """Instantiate the model from a params dict.

        Args:
            config: The same params_dict passed throughout the pipeline.

        Returns:
            A ready-to-use model instance (not yet moved to device).
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique string identifier for this model (e.g. 'HRCloudNet')."""
        ...
