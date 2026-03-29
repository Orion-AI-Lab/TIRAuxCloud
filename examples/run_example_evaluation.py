import argparse
import json
import os
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="landsat",
                        choices=["landsat", "landsatMA", "viirs"])
    parser.add_argument("--config", type=str,
                        default="examples/example_saved_model_run.json")
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path) as f:
        cfg = json.load(f)[args.subset]
        
    dataset_dir = Path(cfg["dataset_dir"])
    dataset_folder = Path(cfg["dataset_folder"])
    
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")
    if not dataset_folder.exists():
        raise FileNotFoundError(f"dataset_folder does not exist: {dataset_folder}")
        
    # Call existing script
    cmd = ["python", "model_test.py", "-t", args.subset]
    subprocess.run(cmd, check=True)
    
if __name__ == "__main__":
    main()