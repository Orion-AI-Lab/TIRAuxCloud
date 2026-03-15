import torch
import numpy as np
import os
import pandas as pd
import numpy as np
import torch
from common_metrics import validate_all, record_validation_metrics_to_csv
import sys
parent_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(parent_script_dir)
import gc
from models_tcloud import init_model_and_loaders,get_gradcam_target_layer
from libraries.utils import save_geotiff, get_preds_multi_encoders
from libraries.wandb_retrieve import get_filtered_wandb_runs, wandinit
import json
import argparse
from tqdm import tqdm
from pytorch_grad_cam import GradCAMPlusPlus
import matplotlib.pyplot as plt


class SegmentationTarget:
    """Converts segmentation output to scalar for GradCAM++ backward pass."""
    def __init__(self, class_idx):
        self.class_idx = class_idx
    
    def __call__(self, model_output):
        # SegFormer output is 3D: (batch, height, width) after argmax
        if model_output.dim() == 3:
            return model_output[0].float().sum() 
        else:
            raise ValueError(f"Unexpected model output shape: {model_output.shape}")


def save_inference_images(ibatch, save_inference_dir, results, inputs, outputs, preds, targets, batch_size, test_df, save_logits, num_classes):
    if isinstance(inputs, list):
        nfiles = inputs[0].size(0)
    else:
        nfiles = inputs.size(0)
    for j in range(nfiles):
        idx = ibatch * batch_size + j
        ref_tif = test_df.loc[idx, 'file']
        image_id = os.path.splitext(os.path.basename(ref_tif))[0]

        mask_path = logits_path = None

        # Save class mask
        class_mask = preds[j].numpy().astype(np.uint8)[np.newaxis, ...]
        mask_path = os.path.join(save_inference_dir, f"{image_id}_mask.tif")
        save_geotiff(class_mask, mask_path, ref_tif, dtype="uint8", count=1)

        # Save logits
        if save_logits:
            logits = outputs[j].cpu().numpy().astype(np.float32)
            logits_path = os.path.join(save_inference_dir, f"{image_id}_logits.tif")
            save_geotiff(logits, logits_path, ref_tif, dtype="float32", count=num_classes)

        # Compute per-class IoU
        pred_np = preds[j].numpy().flatten()
        target_np = targets[j].numpy().flatten()
        ious = []
        for c in range(num_classes):
            intersection = np.logical_and(pred_np == c, target_np == c).sum()
            union = np.logical_or(pred_np == c, target_np == c).sum()
            iou = intersection / union if union > 0 else float('nan')
            ious.append(iou)
        mean_iou = np.nanmean(ious)

        results.append({
            "input_file": os.path.basename(ref_tif),
            "mask_file": os.path.basename(mask_path),
            "logits_file": os.path.basename(logits_path) if logits_path else None,
            "mIoU": mean_iou,
            **{f"IoU_{c}": iou for c, iou in enumerate(ious)}
        })

def evaluate_on_test_set(
    params_dict,
    wandbgroup=None,
    project=None):

    save_inference=params_dict.get("save_inference", False)
    save_logits=params_dict.get("save_logits", False)
    
    # Load test set and data

    df = pd.read_csv(os.path.join(params_dict["dataset_folder"],params_dict["dataset"]))
    if params_dict["traintest"]!="test":
        ttest=params_dict["traintest"]
        print(f'############ WARNING! test set parameter is set : {ttest} (not "test") #################')
    test_df = df[df["dataset"] == params_dict["traintest"]].reset_index(drop=True)

    batch_size=params_dict["batch_size"]
    model_path=params_dict["model_file"]
    device=params_dict["device"]
    num_classes=params_dict["num_classes"]

    model,_,test_loader=init_model_and_loaders(params_dict)
    loaded_state_dict = torch.load(model_path, weights_only=True)
    model.load_state_dict(loaded_state_dict)
    model.eval()

    print("Evaluating on test set...")

    if wandbgroup:
        wandbrun=wandinit(params_dict, wandbgroup, project=project)
    else:
        wandbrun=None

    all_preds = []
    all_targets = []
    results=[]

    if save_inference:
        model_file=os.path.basename(model_path)
        save_inference_dir=os.path.join(os.path.dirname(model_path),"inferences",os.path.splitext(model_file)[0])
        params_file=os.path.join(os.path.dirname(model_path),"inferences",os.path.splitext(model_file)[0]+".json")
        with open(params_file, "w") as f:
            json.dump(params_dict, f, indent=4)
        os.makedirs(save_inference_dir, exist_ok=True)
        print(f"Save Inference to: {save_inference_dir}")
    
    gradcam_dir = "landsat_testset_sample/landsat_gradcam_grid_img" # change befor commit 
    os.makedirs(gradcam_dir, exist_ok=True)
    print(f"GradCAM visualizations will be saved to: {gradcam_dir}")

    target_layers = get_gradcam_target_layer(model, params_dict["model_type"])
    gradcam_batch_count = 1 
    gradcam_target_class = 1 
    if params_dict["gradcam_batch_count"] : 
        gradcam_batch_count = params_dict["gradcam_batch_count"]
    if params_dict["gradcam_target_class"] : 
        gradcam_target_class = params_dict["gradcam_target_class"]
    
    # GRAD-CAM++ : This prototype only focuses on Landsat dataset and SegFromer Model 
    for i, (inputs, targets) in enumerate(tqdm(test_loader, desc="Inference Progress")):

        if i < gradcam_batch_count:
            batch_size_actual = inputs.shape[0]
            print(f"Processing all {batch_size_actual} samples in first batch for GradCAM")
            
            for sample_idx in range(batch_size_actual):
                try:
                    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
                        with torch.enable_grad(): # For safety TODO: Check and remove if not required 
                            input_tensor = inputs[sample_idx:sample_idx+1].to(device).float()  
                            input_tensor.requires_grad_(True)
                            targets_for_cam = [SegmentationTarget(gradcam_target_class)] 
                            activation_map = cam(input_tensor=input_tensor, targets=targets_for_cam)
                except RuntimeError as e:
                    print(f"GradCAM error for sample {sample_idx}: {e}")
                    activation_map = None

                if activation_map is not None:
                    
                    all_bands = inputs[sample_idx].cpu().numpy()
                    # TODO : Get this data from params_dict
                    num_display_bands = 3  # First 3 meaningful bands: cloudy_B10_B11, clear_B10_B11, dem
                    
                    # TODO: change this to get from loader  
                    band_names = params_dict.get('features', [f'Band {b+1}' for b in range(num_display_bands)])
                    
                    print(f"Sample {sample_idx}: displaying {num_display_bands} meaningful bands + GradCAM")
                    
                    # Fixed 2x2 grid: [Band0, Band1] [Band2, Heatmap]
                    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
                    axes = axes.flatten()
                    
                    for band_idx in range(num_display_bands):
                        band_data = all_bands[band_idx].astype(float)
                        band_normalized = (band_data - band_data.min()) / (band_data.max() - band_data.min() + 1e-8)
                        axes[band_idx].imshow(band_normalized, cmap='gray')
                        axes[band_idx].set_title(band_names[band_idx], fontsize=12, fontweight='bold')
                        axes[band_idx].axis('off')
                    
                    im = axes[3].imshow(activation_map[0], cmap='jet')
                    axes[3].set_title("GradCAM++ Heatmap (Class 1)", fontsize=12, fontweight='bold')
                    axes[3].axis('off')
                    fig.colorbar(im, ax=axes[3], label='Activation Strength', fraction=0.046, pad=0.04)
                    
                    plt.tight_layout()
                    
                    save_path = os.path.join(gradcam_dir, f"sample_{sample_idx:04d}_gradcam_collage.png")
                    plt.savefig(save_path, dpi=100, bbox_inches='tight')
                    print(f"Saved GradCAM collage to: {save_path}")
                    plt.close()
                else:
                    print(f"GradCAM failed for sample {sample_idx}, no visualization")
            print(f"\n ✓ GradCAM visualization complete for {gradcam_batch_count} batch{ "s" if gradcam_batch_count > 1 else "" }\n")
        with torch.no_grad():
            outputs = get_preds_multi_encoders(model, inputs, device)
        '''
        inputs = inputs.to(device)
        outputs = model(inputs)
        '''
        if isinstance(outputs, tuple):
            preds = torch.argmax(outputs[0], dim=1).cpu()
        else:
            preds = torch.argmax(outputs, dim=1).cpu()

        targets = targets.cpu()
        all_preds.append(preds)
        all_targets.append(targets)

        if save_inference:
            save_inference_images(i, save_inference_dir, results, inputs, outputs, preds, targets, 
                                    batch_size, test_df, save_logits, num_classes)

    if save_inference:
        # Save DataFrame to CSV
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(save_inference_dir, "inference_results.csv")
        results_df.to_csv(csv_path, index=False)

        print(f"Saved inference results to {csv_path}")               

    metrics = validate_all(model, test_loader, params_dict)

    if wandbrun:
        wandbrun.log(metrics)

    #record_validation_metrics_to_csv(os.path.expanduser("~/shared_storage/tcloudDS/benchmarks/test_results_v2.csv"), metrics, params_dict)

    del model
    del test_loader
    torch.cuda.empty_cache()
    gc.collect()

    if wandbrun:
        wandbrun.finish()
    
    return metrics

def main():

    parser = argparse.ArgumentParser(
        description="Evaluation of models on Test set "
    )

    parser.add_argument(
        "-t", "--test_set",
        type=str,
        choices=["viirs", "landsat", "landsatMA"],
        help="Test set identification (viirs, landsat, landsatMA)"
    )

    parser.add_argument(
        "-l", "--list_only",
        action='store_true',
        help='List only matches'
    )

    args = parser.parse_args()
    test_set = args.test_set
    list_only = args.list_only
   
    configfile=os.path.join(script_dir,"configs/saved_models_run.json")
    with open(configfile, 'r') as file:
        configdict = json.load(file)
    if not test_set in configdict:
        print(f"{test_set} : Test set parameters not found")
    configdict=configdict[test_set]

    wdbentity=configdict.get("wdbentity",None)
    wdbproject_source=configdict.get("wdbproject_source", None)
    wdbproject_target=configdict.get("wdbproject_target", None)

    configparams=configdict["config"]

    if "wandb_filter" in configdict:
        filtdict=configdict[f"wandb_filter"]
        wandbgroup=configdict.get("wandb_group", None)

        dfmodels=get_filtered_wandb_runs(wdbentity, wdbproject_source, filtdict)
    
        if len(dfmodels)==0:
            return
    
        print(dfmodels[["Name","config_model_type","Group","config_features","config_num_classes","config_dataset"]])

        if list_only :
            return
    else:
        # if wandb_filter is missing pick all parameters from config
        dfmodels = pd.DataFrame({"a_column": [0]})
        wandbgroup=None

    for index, wandb_row in dfmodels.iterrows():

        paramsdict={}
        #model_path=find_file_recursive(os.path.basename(row["model_file"]), os.path.dirname(row["model_file"]))

        #first pass config params from wandb
        for k in [p for p in dfmodels if p.startswith("config_")]:
            paramsdict[k[7:]]=wandb_row[k] # without config prefix

        #second pass parameters from config files to override
        for k in configparams:
            paramsdict[k]=configparams[k]
        if "config_dataset" in wandb_row and "config_trained" not in wandb_row:
            paramsdict["trained"]=wandb_row["config_dataset"]

        if not "device" in paramsdict:
            paramsdict["device"]="cuda:0"
        # force loading 'test' dataset in case not otherwise configured
        if not "traintest" in configparams:
            paramsdict["traintest"]="test"
    

        print("Starting Test. Model Parameters:")
        print(paramsdict)

        evaluate_on_test_set(
            paramsdict,
            wandbgroup=wandbgroup,
            project=wdbproject_target
        )

if __name__ == "__main__":
    main()
