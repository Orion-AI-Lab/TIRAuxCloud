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
from models_tcloud import init_model_and_loaders
from libraries.utils import save_geotiff, get_preds_multi_encoders
from libraries.wandb_retrieve import get_filtered_wandb_runs, wandinit
import json
import argparse
from tqdm import tqdm

def single_pass_uncertainty(logits):
    probs = torch.softmax(logits, dim=1)
    entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
    return entropy

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
    if save_inference:
        model_file=os.path.basename(model_path)
        save_inference_dir=os.path.join(os.path.dirname(model_path),"inferences",os.path.splitext(model_file)[0])
        params_file=os.path.join(os.path.dirname(model_path),"inferences",os.path.splitext(model_file)[0]+".json")
        with open(params_file, "w") as f:
            json.dump(params_dict, f, indent=4)
        os.makedirs(save_inference_dir, exist_ok=True)
        print(f"Save Inference to: {save_inference_dir}")

    with torch.no_grad():
        total_uncertainty = 0.0
        total_pixels = 0
        results=[]
        for i, (inputs, targets) in enumerate(tqdm(test_loader, desc="Inference Progress")):

            outputs = get_preds_multi_encoders(model, inputs, device)
            '''
            inputs = inputs.to(device)
            outputs = model(inputs)
            '''
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            uncertainty = single_pass_uncertainty(logits).cpu()
            total_uncertainty += uncertainty.sum().item()
            total_pixels += uncertainty.numel()
            if isinstance(outputs, tuple):
                preds = torch.argmax(outputs[0], dim=1).cpu()
            else:
                preds = torch.argmax(outputs, dim=1).cpu()
            uncertainty_scores = outputs.var(dim=0).mean(dim=1)
            print("Uncertainity using py-trco framework : ",uncertainty_scores )
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

    if total_pixels > 0:
        mean_uncertainty = total_uncertainty / total_pixels
    metrics = validate_all(model, test_loader, params_dict,mean_uncertainty)

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
