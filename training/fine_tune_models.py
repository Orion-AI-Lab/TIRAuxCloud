import os
import sys
parent_script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#libraries_path = os.path.join(current_script_dir, '..', 'libraries')
sys.path.append(parent_script_dir)
import wandb
import torch
import os
import gc
from model_builder.models_tcloud import init_model_and_loaders, dict_to_hash_key
import pandas as pd
import segmentation_models_pytorch as smp
import json
import argparse
from libraries.wandb_retrieve import get_filtered_wandb_runs, wandinit
from libraries.utils import set_seed
from model_training import train_model
import torch.nn as nn
from evaluation.model_test import evaluate_on_test_set
import random

def report_frozen_encoder_children(model):
    """Report freeze status for each encoder *child* exactly as shown in .named_children()."""
    if not hasattr(model, "encoder"):
        print("[report] Model has no .encoder"); 
        return
    enc = model.encoder
    frozen, trainable, mixed, noparams = [], [], [], []
    for name, mod in enc.named_children():
        params = list(mod.parameters())
        if not params:
            noparams.append(name)
            continue
        all_frozen = all(not p.requires_grad for p in params)
        all_train = all(p.requires_grad for p in params)
        if all_frozen:
            frozen.append(name)
        elif all_train:
            trainable.append(name)
        else:
            mixed.append(name)
    print(f"[report] Frozen children ({len(frozen)}): {frozen}")
    print(f"[report] Trainable children ({len(trainable)}): {trainable}")
    if mixed:
        print(f"[report] Mixed children ({len(mixed)}): {mixed}")
    if noparams:
        print(f"[report] No-param children ({len(noparams)}): {noparams}")



def summarize_encoder(enc):
    print("\n=== ENCODER SUMMARY ===")
    print("Type:", type(enc).__name__)

    # Top-level children (always modules)
    children = list(enc.named_children())
    print(f"Children ({len(children)}):", [n for n, _ in children][:12])

    # List/tuple attributes and whether they contain modules or metadata
    candidates = []
    for name in dir(enc):
        if name.startswith("_"):
            continue
        try:
            val = getattr(enc, name)
        except Exception:
            continue
        if isinstance(val, (list, tuple)):
            has_mod = any(isinstance(v, nn.Module) for v in val)
            kinds = [type(v).__name__ for v in val[:5]]
            candidates.append((name, len(val), has_mod, kinds))
    if candidates:
        print("\nList/Tuple attributes:")
        for name, ln, has_mod, kinds in candidates:
            tag = "✅ modules" if has_mod else "ℹ️ not modules"
            print(f" - {name}: len={ln}, {tag}, sample types={kinds}")

    # Try callable get_stages(), if present
    if hasattr(enc, "get_stages") and callable(getattr(enc, "get_stages")):
        try:
            gs = list(enc.get_stages())
            print(f"\nget_stages(): len={len(gs)}; types={[type(m).__name__ for m in gs[:6]]}")
        except Exception as e:
            print("get_stages() raised:", repr(e))
    print("=== END ENCODER SUMMARY ===\n")

def freeze_encoder_first_k(model, k: int):
    """Freeze the first k logical stages of the encoder."""
    if not hasattr(model, "encoder") or k <= 0:
        return
    enc = model.encoder

    stages = []

    # ResNet-like encoders: make a 'stem' stage + layer1..layer4
    if all(hasattr(enc, n) for n in ["layer1", "layer2", "layer3", "layer4"]):
        stem_parts = [getattr(enc, n) for n in ["conv1", "bn1", "relu", "maxpool"] if hasattr(enc, n)]
        if stem_parts:
            stages.append(nn.Sequential(*stem_parts))  # stage 0: stem
        stages.extend([getattr(enc, "layer1"), getattr(enc, "layer2"),
                       getattr(enc, "layer3"), getattr(enc, "layer4")])
    else:
        # Generic fallback: ordered children that are modules
        stages = [m for _, m in enc.named_children() if isinstance(m, nn.Module)]

    # Final safety: keep only real modules
    stages = [m for m in stages if isinstance(m, nn.Module)]
    if not stages:
        print("[freeze_encoder_first_k] No encoder stages found to freeze.")
        return

    k = min(k, len(stages))
    for m in stages[:k]:
        for p in m.parameters():
            p.requires_grad = False

    print(f"[freeze_encoder_first_k] Frozen first {k} encoder stages:",
          [type(m).__name__ for m in stages[:k]])

#def models_fine_tune(device, models_csv, MA_input_files, freeze_encoder=None, skip_models_idx=[],wandbrec=True):
def models_fine_tune(params_dict, numruns=1, wandb_group_train=None, wandb_group_test=None, wdbproject=None):
    
    pretrained_model_path=params_dict["model_file"]
    for run in range(1,numruns+1):
        params_dict["lr"]=params_dict["lr"]*params_dict["lr_ratio"]

        params_dict["traintest"]="fine-tune"
        params_dict["run"]=run
        save_dir=params_dict.get("save_dir",False)

        model,train_loader,val_loader=init_model_and_loaders(params_dict)
        loaded_state_dict = torch.load(pretrained_model_path, weights_only=True)
        model.load_state_dict(loaded_state_dict)

        summarize_encoder(model.encoder)

        freeze_all = bool(params_dict.get("freeze_encoder", False))
        freeze_first_k = int(params_dict.get("freeze_encoder_first_k", 0))

        if freeze_all and hasattr(model, "encoder"):
            for p in model.encoder.parameters():
                p.requires_grad = False            
        elif freeze_first_k > 0:
            freeze_encoder_first_k(model, freeze_first_k)
            
        report_frozen_encoder_children(model)

        modelhashname=dict_to_hash_key(params_dict)
        new_modelfile=None
        if save_dir:
            new_modelfile=os.path.join(save_dir,"model_hash_"+modelhashname+".pth")
            params_dict["model_file"]=new_modelfile
        else:
            del params_dict["model_file"]

        print(f"Fine Tune Train #: {run}")
        print(params_dict)

        if wandb_group_train:
            wandbrun=wandinit(params_dict, wandb_group_train, project=wdbproject)
        else:
            wandbrun=None
        
        train_model(
            model, 
            train_loader,
            val_loader,
            params_dict,
            save_dir=save_dir,
            wandbrun=wandbrun,
        )

        if wandbrun:
            wandbrun.finish()

        del model
        del train_loader
        del val_loader
        torch.cuda.empty_cache()
        gc.collect()

        if new_modelfile:

            params_dict["traintest"]="test"
            print(params_dict)
            
            evaluate_on_test_set(params_dict, wandbgroup=wandb_group_test, project=wdbproject)

def main():

    parser = argparse.ArgumentParser(
        description="Fine tune models"
    )

    parser.add_argument(
        "-t", "--test_set",
        type=str,
        default="viirs",
        choices=["viirs", "landsatMA", "forest2"],
        help="Test set identification (viirs, landsatMA, forest2)"
    )

    parser.add_argument(
        "-l", "--list_only",
        action='store_true',
        help='List only matches'
    )

    args = parser.parse_args()
    test_set = args.test_set
    list_only = args.list_only

    configfile=os.path.join(parent_script_dir,"DSv3/configs/fine_tune_params.json")
    with open(configfile, 'r') as file:
        configdict = json.load(file)
    if not test_set in configdict:
        print(f"{test_set} : Test set parameters not found")
    configdict=configdict[test_set]
    filtdict=configdict[f"wandb_filter"]

    wdbentity=configdict.get("wdbentity",None)
    wdbproject_source=configdict.get("wdbproject_source",None)
    wdbproject_target=configdict.get("wdbproject_target",None)
    wandbgroup=configdict.get("wandb_group",None)
    wandbgroup_test=configdict.get("wandb_group_test",None)
    
    configparams=configdict["config"]

    dffilt=get_filtered_wandb_runs(wdbentity, wdbproject_source, filtdict)
    if len(dffilt)==0:
        return

    print(dffilt[["Name","config_model_type","Group","config_features","config_num_classes","config_dataset"]])
    if list_only:
        return
    
    for index, row in dffilt.iterrows():        
        paramsdict={}
        #model_path=find_file_recursive(os.path.basename(row["model_file"]), os.path.dirname(row["model_file"]))

        #first pass config params from wandb
        for k in [p for p in dffilt if p.startswith("config_")]:
            paramsdict[k[7:]]=row[k]

        #second pass parameters from config files to override
        for k in configparams:
            paramsdict[k]=configparams[k]

        paramsdict["trained"]=row["config_dataset"]
        runs=configparams.get("runs",1)
        if not "device" in paramsdict:
            paramsdict["device"]="cuda:0"
        
        models_fine_tune(
            paramsdict,
            numruns=runs,
            wandb_group_train=wandbgroup,
            wandb_group_test=wandbgroup_test,
            wdbproject=wdbproject_target
        )

if __name__ == "__main__":
    main()

