from typing import Dict, Type, Callable
import torch.nn as nn

_LOSS_REGISTRY: Dict[str, Callable] = {}

def register_loss(name: str):
    def decorator(fn):
        _LOSS_REGISTRY[name] = fn
        return fn
    return decorator

def get_loss(name: str, class_counts=None, device=None) -> nn.Module:
    if name not in _LOSS_REGISTRY:
        raise ValueError(f"Unknown loss '{name}'. Available: {list(_LOSS_REGISTRY)}")
    return _LOSS_REGISTRY[name](class_counts=class_counts, device=device)