from typing import Dict, Type
from model_builder.base_model import BaseModel

_MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {}

def register_model(name: str):
    """Decorator to register a model class by name."""
    def decorator(cls: Type[BaseModel]):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model(name: str, config: dict) -> BaseModel:
    if name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name].from_config(config)

def list_models():
    return list(_MODEL_REGISTRY.keys())