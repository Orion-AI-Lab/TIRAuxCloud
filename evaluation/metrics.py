import numpy as np

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
