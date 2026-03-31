import torch
import os
import hashlib
import json
from data.loaders import get_loaders
from model_builder.registry import get_model, list_models

def find_file_recursive(filename, search_dir):
    """
    Recursively search for a file by name under a directory.

    Args:
        filename (str): Name of the file to search for.
        search_dir (str): Directory path to start the search from.

    Returns:
        list of str: Full paths to files that match the filename.
    """
 
    for root, dirs, files in os.walk(search_dir):
        if filename in files:
            return os.path.join(root, filename)

def sort_lists(obj):
    """
    Recursively sort all lists in the object.
    """
    if isinstance(obj, dict):
        return {k: sort_lists(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, list):
        return sorted((sort_lists(item) for item in obj), key=lambda x: json.dumps(x, sort_keys=True))
    else:
        return obj

def dict_to_hash_key(d, length=15):
    """
    Serialize a dictionary deterministically and hash it to a short key.
    
    Args:
        d (dict): Input dictionary.
        length (int): Desired length of output hash string.
    
    Returns:
        str: A short deterministic hash string.
    """ 
    sorted_obj = sort_lists(d)
    json_str = json.dumps(sorted_obj, sort_keys=True, separators=(',', ':'))
    #print(json_str)
    hash_digest = hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    return hash_digest[:length]

def save_model_and_log_params(model, save_dir, model_file):
    os.makedirs(save_dir, exist_ok=True)

    #model_filename = f"model_{run_id}.pth"

    model_path = os.path.join(save_dir, model_file)
    
    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path}")

    # Prepare CSV log path
    '''
    csv_log_path = os.path.join(save_dir, "model_params_log.csv")
    #params_dict_fname = model_file.copy()
    #params_dict_fname["model_filename"] = model_file

    # Check for duplicates and write log
    if os.path.exists(csv_log_path):
        df = pd.read_csv(csv_log_path)
        if model_file in df["model_filename"].values:
            return model_path
        df = pd.concat([df, pd.DataFrame([params_dict_fname])], ignore_index=True)
    else:
        df = pd.DataFrame([params_dict_fname])

    df.to_csv(csv_log_path, index=False)
    '''
    return model_path

def tolist(item):
    if item is None:
        return None
    elif isinstance(item, list):
       return item
    elif isinstance(item, str):
       return [item]
    
def get_features_two_enc(featset):
    if isinstance(featset[0],str):
        return [featset[0]], [featset[1]]
    elif isinstance(featset[0],list):
        return featset[0], featset[1]
    

def init_model_and_loaders(params_dict, onlyloaders=False):
    featset=params_dict["features"]
    featset=tolist(featset)
    model_type=params_dict["model_type"]
    thincloudcl=params_dict.get("thin_cloud_class",1)
    transformkey=params_dict.get("transform",None)
    batch_size=params_dict["batch_size"]
    input_files = os.path.join(params_dict["dataset_folder"],params_dict["dataset"])
    device=params_dict["device"]
    test=(params_dict["traintest"]=="test")
    target_band=params_dict["target_band"]
    workers=params_dict.get("cpuworkers",4)
    yshift=params_dict.get("yshift",1)
    dataset_dir=params_dict.get("dataset_dir",None)
    
    initmodel=None

    one_enc_models=["Unet","SegFormer","DeepLabV3","Swin-Unet","HRCloudNet","CDnetV2","SwinCloud"]
    one_enc_models=["Fine Tune "+bm for bm in one_enc_models]+one_enc_models
    two_enc_models=["Siamese", "bam-cd"]
    two_enc_models=["Fine Tune "+bm for bm in two_enc_models]+two_enc_models
    
    
    if model_type in one_enc_models:
        clear_bands=None
    elif model_type in two_enc_models:
        featset, clear_bands = get_features_two_enc(featset)

    train_loader, val_loader = get_loaders(input_files, None, featset, target_band, yshift=yshift, 
                                                   clear_bands=clear_bands, batch_size=batch_size, thincloudcl=thincloudcl, 
                                                   transformkey=transformkey, model_type=model_type, testrun=test,
                                                   dataset_dir=dataset_dir, workers=workers)

    
    if onlyloaders:
        return None, train_loader, val_loader
    if model_type in list_models() : 
            initmodel = get_model(model_type, params_dict).to(device)   
    else:
        print(f"Unrecognized Model Type: {model_type}")
        return None, None, None

    return initmodel, train_loader, val_loader
