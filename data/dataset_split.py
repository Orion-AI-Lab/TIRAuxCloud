import os
import re
import random
from collections import defaultdict
from glob import glob
import pandas as pd
import argparse

def extract_scene_id(filename, pattern):
    """Extract Landsat scene ID (e.g. LC09_L1TP_232064) from filename."""
    match = re.search(pattern, filename)
    #match = re.match(r"(LC08|LC09)_L1TP_\d{6}", filename)
    return match.group(0) if match else None

def group_files_by_scene(filenames, pattern):
    """Group file paths by scene ID."""
    scene_to_files = defaultdict(list)
    for f in filenames:
        scene_id = extract_scene_id(f, pattern)
        #scene_id = os.path.basename(f)[10:16]
        if scene_id:
            scene_to_files[scene_id].append(f)
        else:
            print(f"Unmatched File {f}")
    return scene_to_files

def write_split_to_csv(train_files, val_files, test_files, out_path="dataset_split.csv"):
    all_data = []

    for f in train_files:
        all_data.append({"file": f, "dataset": "train"})
    for f in val_files:
        all_data.append({"file": f, "dataset": "val"})
    for f in test_files:
        all_data.append({"file": f, "dataset": "test"})

    df = pd.DataFrame(all_data)
    df.to_csv(out_path, index=False)
    print(f"✅ Dataset split written to {out_path}")

def custom_group_split(file_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42, pattern=r"(LC08|LC09)_L1.._\d{6}"):
    #assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Splits must sum to 1.0"

    scene_to_files = group_files_by_scene(file_list, pattern)
    scene_ids = list(scene_to_files.keys())

    random.seed(seed)
    random.shuffle(scene_ids)

    n_total = len(scene_ids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = int(test_ratio * n_total)
    sum_scenes = n_train + n_val + n_test
    if sum_scenes < n_total and abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        sum_scenes = n_total

    train_scenes = scene_ids[:n_train]
    val_scenes = scene_ids[n_train:n_train + n_val]
    test_scenes = scene_ids[n_train + n_val:sum_scenes]

    # Gather files from each group
    train_files = [f for sid in train_scenes for f in scene_to_files[sid]]
    val_files = [f for sid in val_scenes for f in scene_to_files[sid]]
    test_files = [f for sid in test_scenes for f in scene_to_files[sid]]

    return train_files, val_files, test_files

def regular_split(file_list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    
    random.seed(seed)
    random.shuffle(file_list)

    n_total = len(file_list)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = int(test_ratio * n_total)
    sum_scenes = n_train + n_val + n_test
    if sum_scenes < n_total and abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6:
        sum_scenes = n_total

    train_files = file_list[:n_train]
    val_files = file_list[n_train:n_train + n_val]
    test_files = file_list[n_train + n_val:sum_scenes]

    return train_files, val_files, test_files


def main():
    parser = argparse.ArgumentParser(
        description="Split a list of file paths into training, validation, and test sets."
    )

    parser.add_argument(
        "-i", "--input_folder",
        type=str,
        required=True,
        help="Path to the folder containing the images"
    )
    parser.add_argument(
        "-s", "--split_type", 
        type=str,
        default="random",
        choices=["random","scene_group"],
        help="Type of data split (random, scene_group)"
    )
    parser.add_argument(
        "-o", "--output_base_name", # Short parameter name added here
        type=str,
        required=True,
        help="Base name for the output split list files (e.g., 'my_dataset_split'). "
             "This will generate files like 'my_dataset_split_train.txt', "
             "'my_dataset_split_val.txt', 'my_dataset_split_test.txt'."
    )
    parser.add_argument(
        "-trr","--train_ratio",
        type=float,
        default=0.7,
        help="Ratio of data for the training set (e.g., 0.7 for 70%%)."
    )
    parser.add_argument(
        "-vr","--val_ratio",
        type=float,
        default=0.15,
        help="Ratio of data for the validation set (e.g., 0.15 for 15%%)."
    )
    parser.add_argument(
        "-ter","--test_ratio",
        type=float,
        default=0.15,
        help="Ratio of data for the test set (e.g., 0.15 for 15%%)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42, # A common default seed for reproducibility
        help="Random seed for reproducibility of the split."
    )
    parser.add_argument(
        "-p","--pattern",
        type=str,
        default=r"[0-9a-f]{64}_(IR1|IR2)", 
        # A regex pattern to select the scenes unique names 
        # forset2: r"[0-9a-f]{64}_(IR1|IR2)", landsat r"(LC08|LC09)_L1.._\d{6}")
        help="Scenes file pattern"
    )

    args = parser.parse_args()

    # Derive the full output file names from the base name
    input_folder = args.input_folder
    split_type = args.split_type
    split_list = args.output_base_name
    train_ratio=args.train_ratio
    val_ratio=args.val_ratio
    test_ratio=args.test_ratio
    seed=args.seed
    pattern=args.pattern

    #image_files = glob(os.path.expanduser("~/shared_storage/tcloudDS/patches_v3MA_0717/m100_p256/allnorm/*tif"))
    image_files = glob(os.path.expanduser(input_folder)+"/*tif")

    if split_type=="scene_group":
        # last run train_ratio=0.40, val_ratio=0.25, test_ratio=0.35, seed=42 test_dataset_split_0719.csv
        train_files, val_files, test_files = custom_group_split(image_files, train_ratio=train_ratio, val_ratio=val_ratio, 
                                                                test_ratio=test_ratio, seed=seed, pattern=pattern)
    elif split_type=="random":
        train_files, val_files, test_files = regular_split(image_files, train_ratio=train_ratio, val_ratio=val_ratio, 
                                                           test_ratio=test_ratio, seed=seed)
    #write_split_to_csv(train_files, val_files, test_files, 
    # out_path=os.path.expanduser('~/nvme1/usgslandsat/DSv3/training_file_lists/test_dataset_split_0719.csv'))
    
    write_split_to_csv(train_files, val_files, test_files, out_path=os.path.expanduser(split_list))

    sumf = len(train_files) + len(val_files) + len(test_files)

    print(f"Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}, Sum :{sumf}")

if __name__ == "__main__":
    main()

'''
python dataset_split.py 
  -i ~/shared_storage/tcloudDS/viirs_patches/m200_p256/allnorm 
  -s random 
  -o ~/nvme1/usgslandsat/DSv3/training_file_lists/split_viirs.csv 
  -trr 0.3 -vr 0.2 -ter 0.5

python dataset_split.py 
  -i /mnt/shared_storage/tcloudDS/patches_CVPRMA/m100_p256/allnorm/
  -s scene_group
  -o /mnt/nvme1/usgslandsat/DSv3/training_file_lists/split_testtest_strict.csv
  -trr 0.6 -vr 0.2 -ter 0.2 -p '(LC08|LC09)_L1.._\d{6}'
'''

