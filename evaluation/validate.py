import torch
from tqdm import tqdm
import os
import pandas as pd
from training.loss_registry import get_loss
from evaluation.metrics import calculate_metrics

def validate_all(model, val_loader, params_dict):
    """
    Validation function for single-head U-Net models.
    Expects data loader to return (x, y) format.
    """
    model.eval()
    total_pixels = 0
    correct_pixels = 0
    all_preds = []
    all_targets = []
    device = params_dict["device"]

    if "loss" in params_dict:
        criterion = get_loss(params_dict["loss"], params_dict.get("class_counts",None), device)
    
    total_loss=0
    total_batches=0

    with torch.no_grad():
        for x, y in tqdm(val_loader, desc="Validating", leave=False):
            y = y.to(device)
            # Move to device
            if isinstance(x, tuple) or isinstance(x, list):
                cloudy, clear = x
                cloudy, clear = cloudy.to(device), clear.to(device)
                logits = model(cloudy, clear)
            else:
                x = x.to(device)
                logits = model(x)

            if isinstance(logits, tuple):
                preds = logits[0].argmax(dim=1)
                if "loss" in params_dict:
                    loss = criterion(logits[0],logits[1],y)
            else:
                preds = logits.argmax(dim=1)
                if "loss" in params_dict:
                    loss = criterion(logits, y)
 
            if "loss" in params_dict:
                total_loss += loss.item()
                total_batches += 1
            
            # Accumulate accuracy
            correct_pixels += (preds == y).sum().item()
            total_pixels += y.numel()
            
            # Store predictions and targets
            preds_np = preds.view(-1).cpu().numpy()
            y_np = y.view(-1).cpu().numpy()
            all_preds.extend(preds_np)
            all_targets.extend(y_np)
    
    if "loss" in params_dict:
        avg_loss = total_loss / total_batches

    # Calculate metrics
    metrics = calculate_metrics(all_preds, all_targets, params_dict["num_classes"], total_pixels, correct_pixels)
    if "loss" in params_dict:
        metrics["val_loss"] = avg_loss
        metrics["neg_val_loss"] = -avg_loss
        print(f"Val. Loss: {avg_loss}")
    
    # Calculate all metrics using shared function
    return metrics

def record_validation_metrics_to_csv(csv_path, metrics_dict, params_dict, wandbrun=None):
    row = {}
    row.update(params_dict)
    row.update(metrics_dict)
    
    if wandbrun:
        for k in metrics_dict:
            wandbrun.summary[k]=metrics_dict[k]

    df = pd.DataFrame([row])

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)
