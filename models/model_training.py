import os
import sys
parent_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#libraries_path = os.path.join(current_script_dir, '..', 'libraries')
sys.path.append(parent_script_dir)
import wandb
import torch
import numpy as np
from tqdm import tqdm
import os
import numpy as np
import albumentations as A
from tqdm import tqdm
import gc
from common_metrics import validate_all, record_validation_metrics_to_csv, getLossFunction
from models_tcloud import save_model_and_log_params, dict_to_hash_key,init_model_and_loaders
import torch.nn.functional as F
import pandas as pd
import segmentation_models_pytorch as smp
from libraries.utils import save_geotiff, write_dict_to_json
import json
from model_test import evaluate_on_test_set
from libraries.wandb_retrieve import wandinit
from libraries.utils import get_preds_multi_encoders, set_seed
import random

def early_stop(model, early_stop_dict, params_dict, save_dir=None, wandbrun=None):
    
    # Early stopping logic
    if early_stop_dict["early_stop_metric"] > early_stop_dict["best_early_stop"]:
        early_stop_dict["best_early_stop"] = early_stop_dict["early_stop_metric"]
        early_stop_dict["epochs_no_improve"] = 0
        #torch.save(model.state_dict(), model_save_path)
        if save_dir:
            model_path=save_model_and_log_params(model, save_dir, params_dict["model_file"])
            params_dict["model_file"]=model_path
        best_stop=early_stop_dict["best_early_stop"]
        target=params_dict["target_metric"]
        mess="saved" if save_dir else ""
        print(f"  New best model {mess} ({target}={best_stop:.4f})")
        early_stop_dict["best_metrics"] = early_stop_dict["metrics"].copy()
    else:
        early_stop_dict["epochs_no_improve"] += 1
        if early_stop_dict["epochs_no_improve"] >= early_stop_dict["patience"]:
            epochs_no_improve=early_stop_dict["epochs_no_improve"]
            print(f"Early stopping triggered after {epochs_no_improve} epochs with no improvement.")
            if params_dict["results_csv"]:
                record_validation_metrics_to_csv(params_dict["results_csv"], early_stop_dict["best_metrics"], params_dict, wandbrun=wandbrun)
            return early_stop_dict, True
    return early_stop_dict, False

def get_optimizer(params_dict, model):
    if params_dict.get("optimizer") == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params_dict["lr"],
            weight_decay=params_dict.get("weight_decay", 0.0)
        )
    else:
        # default Adam
        if "weight_decay" in params_dict:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params_dict["lr"],
                weight_decay=params_dict["weight_decay"]
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params_dict["lr"]
            )
    return optimizer

def train_model(
    model, 
    train_loader,
    val_loader,
    params_dict,
    save_dir=False,
    wandbrun=None,
):

    device=params_dict["device"]
    seed = params_dict.get("seed",random.randint(0, 2**32 - 1))
    set_seed(seed)
    params_dict["seed"]=seed

    loss_fn = getLossFunction(params_dict["loss"], params_dict.get("class_counts",None), device)

    optimizer = get_optimizer(params_dict, model)
    
    patience=params_dict["patience"]
    early_stop_dict={
      "best_early_stop" : 0,
      "epochs_no_improve" : 0,
      "patience":patience
    }
    first_epoch=True
    max_epochs = params_dict.get("max_epochs", 200)
    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        print("\n")
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs} - Training"):
            y = y.to(device)
            '''
            if isinstance(x, tuple) or isinstance(x, list):
                cloudy, clear = x
                cloudy = cloudy.to(device)
                clear = clear.to(device)
                preds = model(cloudy, clear)
            else:
                x = x.to(device)
                preds = model(x)
            '''
            preds = get_preds_multi_encoders(model, x, device)

            if isinstance(preds, tuple):
                loss = loss_fn(preds[0],preds[1],y)
            else:
                loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        print(f"Train loss : {avg_train_loss}")

        metrics = validate_all(model, val_loader, params_dict)
        metrics["train_loss"]=avg_train_loss
        metrics["epochs_best"]=epoch

        if first_epoch:
            early_stop_dict["best_early_stop"]=metrics[params_dict["target_metric"]]-1

        early_stop_dict["metrics"]=metrics
        early_stop_dict["early_stop_metric"]=metrics[params_dict["target_metric"]]
        early_stop_dict["epoch"]=epoch

        if wandbrun:
            wandbrun.log(metrics)

        early_stop_dict, stop = early_stop(model,early_stop_dict, params_dict, save_dir, wandbrun=wandbrun)

        first_epoch=False

        if stop:
            break
    beststop=early_stop_dict["best_early_stop"]
    target=params_dict["target_metric"]
    print(f"Training finished. Best {target}: {beststop:.4f}")

#device = torch.device("cuda:" if torch.cuda.is_available() else "cpu")

def models_training(paramsrun):
    
    runs = paramsrun["runs"] if "runs" in paramsrun else 1
    for loss_t in paramsrun["losslist"]:
        for model_dict in paramsrun["modellist"]:
            for i, featset in enumerate(paramsrun["feature_sets"]):
                for class_dict in paramsrun["classlist"]:
                #for transform in [None,"full"]:

                    #filter: in cases we stop the training change minor parameters and don't want to retrain some model combinations 
                    # if model_item["Name"]=="Unet" and loss_t=="CrossEntropy" and \
                    # featset==["cloudy_B"+str(b) for b in [1, 2, 3, 4, 6, 7, 9]]+["cloudy_Radiance_B10_B11_mean"]:
                    #     print("Model in filtered out combination, Skipping!")
                    #     continue

                    for run_count in range(1,runs+1):
                        paramsdict={}
                        paramsdict["run"]=run_count
                        paramsdict["patience"]=paramsrun["patience"]
                        paramsdict["loss"]=loss_t
                        for k in class_dict:
                            paramsdict[k]=class_dict[k]
                        for k in model_dict:
                            paramsdict[k]=model_dict[k]
                        paramsdict["device"]=paramsrun["device"]
                        paramsdict["dataset"]=paramsrun["dataset"]
                        paramsdict["target_metric"]=paramsrun["target_metric"]
                        paramsdict["target_band"]=paramsrun["target_band"]
                        paramsdict["features"]=featset
                        paramsdict["results_csv"]=os.path.expanduser(paramsrun["results_csv"])
                        paramsdict["dataset_folder"]=os.path.expanduser(paramsrun["dataset_folder"])
                        paramsdict["transform"]=paramsrun.get("transform")
                        paramsdict["traintest"]="train"
                        paramsdict["cpuworkers"]=paramsrun["cpuworkers"]
                        if "yshift" in paramsrun:
                            paramsdict["yshift"]=paramsrun.get("yshift")
                        if "fusion_mode" in paramsrun:
                            paramsdict["fusion_mode"]=paramsrun["fusion_mode"]
                        if "dataset_dir" in paramsrun:
                            paramsdict["dataset_dir"]=paramsrun["dataset_dir"]
                        if "max_epochs" in paramsrun:
                            paramsdict["max_epochs"]=paramsrun["max_epochs"]
                        paramsdict

                        wdbproject = paramsrun.get("wdbproject", None)
                        wdbentity = paramsrun.get("wdbentity", None)

                        print(f"Starting Train : {model_dict}")
                        print(paramsdict)

                        save_dir=os.path.expanduser(paramsrun["save_dir"])
                        modelhashname=dict_to_hash_key(paramsdict)
                        _modelfile=os.path.join(save_dir,"model_hash_"+modelhashname+".pth")
                        if paramsrun["save_model"]: paramsdict["model_file"]=_modelfile

                        if not os.path.isfile(_modelfile) or paramsrun["save_model"]=="force":
    
                            if "wandb_train_group" in paramsrun:
                                wandbrun=wandinit(paramsdict, paramsrun["wandb_train_group"],
                                                  entity=wdbentity, project=wdbproject)
                            else:
                                wandbrun=None

                            model,train_loader,val_loader=init_model_and_loaders(paramsdict)
                            if model is None: continue

                            sav=save_dir if paramsrun["save_model"] else paramsrun["save_model"]
                
                            train_model(
                                model=model,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                params_dict=paramsdict,
                                #save_dir=None,
                                save_dir=sav,
                                wandbrun=wandbrun
                            )
                        
                            del model
                            del train_loader
                            del val_loader
                            torch.cuda.empty_cache()
                            gc.collect()

                            if wandbrun:
                                wandbrun.finish()
                        else:
                            paramsdict["model_file"]=_modelfile
                            print("Existing model file, skipping training and starting only inference on test data")

                        if "model_file" in paramsdict:

                            paramsdict["traintest"]="test"
                            print(f"Starting Test : {model_dict}")
                            print(paramsdict)

                            if "wandb_test_group" in paramsrun:
                                wandbgroup=paramsrun["wandb_test_group"]
                            else:
                                wandbgroup=None
                            
                            evaluate_on_test_set(paramsdict, wandbgroup=wandbgroup, project=wdbproject)


def main():

    run_json=os.path.join(parent_script_dir,"DSv3/configs/train_params_run.json")
    with open(run_json, 'r') as file:
        paramsrun = json.load(file)

    models_training(paramsrun)
    

if __name__ == '__main__':
    main()

