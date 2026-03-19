import torch
import numpy as np
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import albumentations as A
from training.loss import getLossFunction

def iou_score(preds, targets, num_classes=2):
    preds = preds.view(-1)
    targets = targets.view(-1)
    ious = []
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = targets == cls
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(float('nan'))  # Ignore this class
        else:
            ious.append(intersection / union)
    # Return mean IoU ignoring NaNs
    return np.nanmean(ious)

def iou_per_class(preds, labels, num_classes=3):
    preds = preds.detach().cpu()
    labels = labels.detach().cpu()
    ious = []

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        label_inds = (labels == cls)

        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()

        if union == 0:
            iou = float('nan')  # or 1.0 if you prefer
        else:
            iou = intersection / union
        ious.append(iou)
    return ious

def calculate_metrics(all_preds, all_targets, num_classes, total_pixels, correct_pixels):
    """
    Calculate validation metrics from predictions and targets.
    
    Args:
        all_preds: List of all predictions (flattened)
        all_targets: List of all targets (flattened)
        num_classes: Number of classes
        total_pixels: Total number of pixels
        correct_pixels: Number of correctly predicted pixels
    
    Returns:
        target_metric: Average IoU for early stopping
        metrics: Dictionary of all metrics
    """
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Store confusion matrix components for each class
    tp = np.zeros(num_classes, dtype=np.uint64)
    fp = np.zeros(num_classes, dtype=np.uint64)
    fn = np.zeros(num_classes, dtype=np.uint64)
    tn = np.zeros(num_classes, dtype=np.uint64)
    
    # Calculate per-class metrics
    for cls in range(num_classes):
        cls_pred = (all_preds == cls)
        cls_true = (all_targets == cls)
        tp[cls] = np.logical_and(cls_pred, cls_true).sum()
        fp[cls] = np.logical_and(cls_pred, ~cls_true).sum()
        fn[cls] = np.logical_and(~cls_pred, cls_true).sum()
        tn[cls] = np.logical_and(~cls_pred, ~cls_true).sum()
    
    # Compute metrics
    epsilon = 1e-7
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)
    pixel_accuracy = correct_pixels / total_pixels
    
    # Confusion matrix
    # cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    
    # Build metrics dictionary
    metrics = {
        "pixel_accuracy": pixel_accuracy,
    }
    avg_iou = 0
    for cls in range(num_classes):
        metrics[f"iou_{cls}"] = iou[cls]
        metrics[f"precision_{cls}"] = precision[cls]
        metrics[f"recall_{cls}"] = recall[cls]
        metrics[f"f1_{cls}"] = f1[cls]
        metrics[f"tp_{cls}"] = tp[cls]
        metrics[f"fn_{cls}"] = fn[cls]
        metrics[f"tn_{cls}"] = tn[cls]
        metrics[f"fp_{cls}"] = fp[cls]
        avg_iou += iou[cls]
  
    avg_iou = avg_iou / num_classes
    metrics[f"iou_avg"]=avg_iou

    # metrics["confusion_matrix"] = cm.flatten().tolist()

     # Print results
    print(f"\nPixel Accuracy: {pixel_accuracy:.4f}, mIoU: {avg_iou}")
    print(f"{'Class':<6} {'IoU':>6} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    for cls in range(num_classes):
        print(f"{cls:<6} {iou[cls]:>6.3f} {precision[cls]:>10.3f} {recall[cls]:>8.3f} {f1[cls]:>6.3f}")
   
    return metrics

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
        criterion = getLossFunction(params_dict["loss"], params_dict.get("class_counts",None), device)
    
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
