from pathlib import Path
import rasterio
from rasterio.transform import Affine
from rasterio.enums import Resampling
import json
import yaml
import os
import random
import numpy as np
import torch

def find_file_with_string(folder, substring):
    folder_path = Path(folder)
    for file in folder_path.iterdir():
        if file.is_file() and substring in file.name:
            return file.name, file.suffix  # filename without extension, and extension
    return None, None  # If not found


def save_geotiff(data, out_path, reference_path, dtype, count):
    """Save array as GeoTIFF using reference image metadata."""
    with rasterio.open(os.path.expanduser(reference_path)) as ref:
        meta = ref.meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": data.shape[1],
            "width": data.shape[2],
            "count": count,
            "dtype": dtype,
            "crs": ref.crs,
            "transform": ref.transform
        })
        with rasterio.open(out_path, 'w', **meta) as dst:
            if count == 1:
                dst.write(data[0], 1)
            else:
                dst.write(data)

def write_dict_to_yaml(data_dict, file_path):
    """
    Writes a Python dictionary to a YAML file.

    Args:
        data_dict (dict): The dictionary to write.
        file_path (str): The path to the output YAML file (e.g., 'output_config.yaml').
    """
    try:
        with open(file_path, 'w') as file:
            yaml.dump(data_dict, file, default_flow_style=False, sort_keys=False)
        print(f"Dictionary successfully written to '{file_path}'")
    except Exception as e:
        print(f"Error writing dictionary to YAML: {e}")

def write_dict_to_json(data_dict, file_path, indent=4):
    """
    Writes a Python dictionary to a JSON file.

    Args:
        data_dict (dict): The dictionary to write.
        file_path (str): The path to the output JSON file (e.g., 'output_config.json').
        indent (int, optional): The number of spaces to use for indentation.
                                Set to None for a compact JSON output. Defaults to 4.
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(data_dict, file, indent=indent)
        print(f"Dictionary successfully written to '{file_path}'")
    except Exception as e:
        print(f"Error writing dictionary to JSON: {e}")

def get_preds_multi_encoders(model, inputs, device):
    if isinstance(inputs, tuple) or isinstance(inputs, list):
        allinps=[]
        for x in inputs:
            x=x.to(device)
            allinps.append(x)
        preds = model(*allinps)
    else:
        x = inputs.to(device)
        preds = model(x)
    return preds

def set_seed(seed: int):
    """Set all relevant seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)              # CPU
    torch.cuda.manual_seed(seed)         # current GPU
    torch.cuda.manual_seed_all(seed)     # all GPUs

    # Determinism settings (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Optional strict determinism (raises on nondet ops)
    # torch.use_deterministic_algorithms(True)