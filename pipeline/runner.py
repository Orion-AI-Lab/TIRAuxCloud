from typing import List, Optional

from model_builder.registry import get_model
from training.loss_registry import get_loss
from training.hooks import TrainingHook, EntropyRegHook, UncertaintyHook
from training.cloud_trainer import CloudTrainer
from training.model_training import get_optimizer


def build_pipeline(
    params_dict: dict,
    hooks: Optional[List[TrainingHook]] = None,
) -> CloudTrainer:
    """
    Assembles model + optimizer + loss_fn + hooks into a CloudTrainer.

    Args:
        params_dict: The standard config dict used throughout the pipeline.
                     Must contain at minimum:
                       model_type, features, num_classes, device,
                       loss, lr, optimizer, patience
        hooks:       List of TrainingHook instances to attach.
                     Defaults to [EntropyRegHook(λ=0.1), UncertaintyHook()].

    Returns:
        CloudTrainer ready to call .train(train_loader, val_loader)
    """
    device = params_dict["device"]

    model = get_model(params_dict["model_type"], params_dict).to(device)
    optimizer = get_optimizer(params_dict, model)
    loss_fn = get_loss(
        params_dict["loss"],
        params_dict.get("class_counts", None),
        device,
    )

    if hooks is None:
        hooks = [
            EntropyRegHook(lambda_reg=params_dict.get("lambda_reg", 0.1)),
            UncertaintyHook(),
        ]

    return CloudTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        params_dict=params_dict,
        hooks=hooks,
    )
