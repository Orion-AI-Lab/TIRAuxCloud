import os
import rasterio
import numpy as np
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import csv
from functools import partial
import traceback
import sys
sys.path.append(os.path.expanduser('~/nvme1/usgslandsat/libraries/'))
from run_parallel import run_in_parallel
import time
#from tdigest import TDigest
import numpy as np
import pandas as pd

def compute_clip_pad_minmax(tif_files, pct=1.0, pad_frac=0.05, bins=4096):
    """
    Compute global min/max for Bands 1–3 using only valid (e.g., cloudy) pixels.
    Clips extreme percentiles (pct%) and pads by pad_frac of the range.
    Returns (mins, maxs) or (None, None) if no valid pixels found.
    """
    import numpy as np, rasterio

    pct = float(pct); pad_frac = float(pad_frac)
    if not (0.0 < pct < 50.0):
        raise ValueError("pct must be in (0, 50).")
    if pad_frac < 0.0:
        raise ValueError("pad_frac must be >= 0.")
    bins = int(bins)

    raw_min = [np.inf, np.inf, np.inf]
    raw_max = [-np.inf, -np.inf, -np.inf]
    any_valid = False

    # Pass 1: find global raw min/max
    for path in tif_files:
        with rasterio.open(path) as src:
            mask = src.read(4)
            valid_mask = (mask == 2) | (mask == 3)  
            if not valid_mask.any():
                continue
            any_valid = True
            for b in range(3):
                arr = src.read(b + 1)
                valid = np.isfinite(arr) & (arr != 0) & valid_mask
                if not valid.any():
                    continue
                vals = arr[valid]
                vmin, vmax = float(vals.min()), float(vals.max())
                raw_min[b] = min(raw_min[b], vmin)
                raw_max[b] = max(raw_max[b], vmax)

    if not any_valid:
        return None, None

    # Pass 2: build histograms within range
    hists = [np.zeros(bins, dtype=np.int64) for _ in range(3)]
    edges_list = [np.linspace(raw_min[b], raw_max[b], bins + 1, dtype=np.float64) for b in range(3)]

    for path in tif_files:
        with rasterio.open(path) as src:
            mask = src.read(4)
            valid_mask = (mask == 2) | (mask == 3)
            if not valid_mask.any():
                continue
            for b in range(3):
                arr = src.read(b + 1)
                valid = np.isfinite(arr) & (arr != 0) & valid_mask
                if not valid.any():
                    continue
                vals = np.clip(arr[valid], raw_min[b], raw_max[b])
                hist, _ = np.histogram(vals, bins=edges_list[b])
                hists[b] += hist

    mins, maxs = [], []
    for b in range(3):
        hist, edges = hists[b], edges_list[b]
        cdf = np.cumsum(hist)
        total = cdf[-1]
        if total == 0:
            return None, None
        lo_rank = pct / 100.0 * total
        hi_rank = (1.0 - pct / 100.0) * total
        lo_idx = int(np.searchsorted(cdf, lo_rank))
        hi_idx = int(np.searchsorted(cdf, hi_rank))
        lo_idx = min(max(lo_idx, 0), len(edges) - 2)
        hi_idx = min(max(hi_idx, 0), len(edges) - 2)
        clipped_lo, clipped_hi = edges[lo_idx], edges[hi_idx + 1]
        if not (clipped_hi > clipped_lo):
            clipped_hi = clipped_lo + 1e-6
        span = clipped_hi - clipped_lo
        pad = pad_frac * span
        mins.append(clipped_lo - pad)
        maxs.append(clipped_hi + pad)

    print(f"Clip+Pad minmax → clip {pct}% → pad {pad_frac}:")
    for i in range(3):
        print(f"  Band {i+1}: min={mins[i]:.6f}, max={maxs[i]:.6f}")
    return mins, maxs


def compute_clip_minmax(tif_files, lower_pct=1.5, upper_pct=98.5, bins=4096):
    """
    Histogram-based robust min/max computation (percentile clipping).
    Faster and more memory-efficient for large datasets.
    """
    import numpy as np, rasterio

    raw_min = [np.inf, np.inf, np.inf]
    raw_max = [-np.inf, -np.inf, -np.inf]
    any_valid = False

    # Pass 1: global min/max bounds
    for path in tif_files:
        with rasterio.open(path) as src:
            for b in range(3):
                arr = src.read(b + 1)
                valid = np.isfinite(arr) & (arr != 0)
                if not valid.any():
                    continue
                vals = arr[valid]
                raw_min[b] = min(raw_min[b], float(vals.min()))
                raw_max[b] = max(raw_max[b], float(vals.max()))
                any_valid = True

    if not any_valid:
        return None, None

    # Pass 2: accumulate histograms
    hists = [np.zeros(bins, dtype=np.int64) for _ in range(3)]
    edges_list = [np.linspace(raw_min[b], raw_max[b], bins + 1, dtype=np.float64) for b in range(3)]

    for path in tif_files:
        with rasterio.open(path) as src:
            for b in range(3):
                arr = src.read(b + 1)
                valid = np.isfinite(arr) & (arr != 0)
                if not valid.any():
                    continue
                vals = np.clip(arr[valid], raw_min[b], raw_max[b])
                hist, _ = np.histogram(vals, bins=edges_list[b])
                hists[b] += hist

    mins, maxs = [], []
    for b in range(3):
        hist, edges = hists[b], edges_list[b]
        cdf = np.cumsum(hist)
        total = cdf[-1]
        if total == 0:
            return None, None
        lo_rank = lower_pct / 100.0 * total
        hi_rank = upper_pct / 100.0 * total
        lo_idx = int(np.searchsorted(cdf, lo_rank))
        hi_idx = int(np.searchsorted(cdf, hi_rank))
        lo_idx = min(max(lo_idx, 0), len(edges) - 2)
        hi_idx = min(max(hi_idx, 0), len(edges) - 2)
        mins.append(float(edges[lo_idx]))
        maxs.append(float(edges[hi_idx + 1]))

    print(f"Clip minmax → {lower_pct:.1f}–{upper_pct:.1f}% (histogram-based):")
    for i in range(3):
        print(f"  Band {i+1}: min={mins[i]:.6f}, max={maxs[i]:.6f}")

    return mins, maxs



def save_stats_to_csv(stats_dict, output_csv_path):
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        categ = next(iter(stats_dict))
        stats = next(iter(stats_dict[categ]))
        stat_labels=[k for k in stats_dict[categ][stats]]
        writer.writerow(['Data Category', 'Band Description']+stat_labels)
        for datacategory, minmaxdicts in stats_dict.items():
            for band_desc, minmaxd in stats_dict[datacategory].items():
                vals = [minmaxd[stat] for stat in stat_labels]
                writer.writerow([datacategory, band_desc]+vals)

def get_band_keys(descriptions, prefix=None):
    keys = []
    for i in range(1, len(descriptions) + 1):
        desc = descriptions[i-1]
        if desc:
            keys.append(desc)
        else:
            keys.append(f"{prefix or 'band'}_{i}")
    return keys

def update_min_max(valid_data, var_dict, bandname):
    if valid_data.size == 0:
        return
    var_dict[bandname]={}
    var_dict[bandname]["min"]=float(np.min(valid_data))
    var_dict[bandname]["max"]=float(np.max(valid_data))

def update_mean_std(chunk, var_dict, bandname):
    if chunk.size == 0:
        return

    sum_ = (chunk*1.0).sum()                
    sumsq = ((chunk*1.0) ** 2).sum()
    count = chunk.shape[0]*1.0             

    var_dict[bandname]['count'] = count
    var_dict[bandname]['sum'] = sum_
    var_dict[bandname]['sumsq'] = sumsq


def update_all(band_data, var_dict, bandnames, debug=False):
    for i, bandname in enumerate(bandnames):
        if debug:
            print(f"Processing Band: {bandname}")
        band = band_data[i]
        valid_data = band[np.isfinite(band)] # filter out avoid tif infinite values
        if not (bandname in ['snow_cover','total_precip_cumul']): # do not filter for total precipitation and snow cover
            valid_data = valid_data[valid_data != 0] # filter tif background 0 values (no-data) - for most bands
        if debug:
            print(f"Processing Min Max: {bandname}")
        update_min_max(valid_data, var_dict, bandname)
        #if debug:
        #    print(f"Processing Histograms: {key}")
        #update_hist(valid_data, var_dict, key)
        if debug:
            print(f"Processing Mean - StD: {bandname}")
        update_mean_std(valid_data, var_dict, bandname)
 
def process_directory(dirpath, debug=False):

    if debug:
        print(f'Processing: {dirpath}')

    filenames = os.listdir(dirpath)

    era5_dict = {}
    clear_cloudy_dict = {}
    
    era5_files = [f for f in filenames if "era5_acqtime" in f and "TIF" in f]
    clear_files = [f for f in filenames if "clear_100" in f]
    cloudy_files = [f for f in filenames if "cloudy_100" in f and "era5" not in f]

    try:
        for f in era5_files:
            filepath = os.path.join(dirpath, f)
            with rasterio.open(filepath) as src:
                if debug:
                    print(f'Start era5: {dirpath}')
                data = src.read()
                keys = get_band_keys(src.descriptions, prefix="era5")
                
                update_all(data, era5_dict, keys)
                if debug:
                    print(f'Finished era5: {dirpath}')

        for clear_file in clear_files:
            #prefix = clear_file.replace("clear.TIF", "")
            #matching_cloudy = [f for f in cloudy_files if f.startswith(prefix)]
            matching_cloudy=[cloudy_files[0]]
            if not matching_cloudy:
                continue

            clear_path = os.path.join(dirpath, clear_file)
            cloudy_path = os.path.join(dirpath, matching_cloudy[0])
            with rasterio.open(clear_path) as clear_src, rasterio.open(cloudy_path) as cloudy_src:
                #assert clear_src.count == cloudy_src.count
                if debug:
                    print(f'Start clear, cloudy: {dirpath}')
                clear_data = clear_src.read()
                cloudy_data = cloudy_src.read()
                # Flatten each band before concatenation
                clear_data_flat = clear_data.reshape(clear_data.shape[0], -1)
                cloudy_data_flat = cloudy_data.reshape(cloudy_data.shape[0], -1)
                combined_data = np.concatenate([clear_data_flat, cloudy_data_flat], axis=1)
                keys = get_band_keys(clear_src.descriptions, prefix="common")
                #update_min_max(combined_data, clear_cloudy_minmax, keys)
                #hist, bin_edges, min_val, max_val, total_count = \
                #update_histogram(combined_data, hist, bin_edges, min_val, max_val, num_bins=1000)
                update_all(combined_data, clear_cloudy_dict, keys)
                if debug:
                    print(f'Finished clear, cloudy: {dirpath}')
    except Exception as e:
        if debug:
            print(f"Failed for folder: {dirpath}")
            traceback.print_exc()
    
    if debug:
        print(f'Finished: {dirpath}')

    return era5_dict, clear_cloudy_dict

def merge_dicts(dicts):
    merged = {}
    for d in dicts:
        if len(d) == 0:
            continue
        for bandname, val in d.items():
            if bandname not in merged:
                merged[bandname] = val
            else:
                if "min" in val and "max" in val:
                    merged[bandname]["min"] = min(merged[bandname]["min"], val["min"])
                    merged[bandname]["max"] = max(merged[bandname]["max"], val["max"])
                if "count" in val and "sum" in val and "sumsq" in val:
                    merged[bandname]["count"] += val["count"]
                    merged[bandname]["sum"] += val["sum"]
                    merged[bandname]["sumsq"] += val["sumsq"]
    for bandname, val in merged.items():
        if "count" in val and "sum" in val and "sumsq" in val:
            if merged[bandname]["count"]>0:
                merged[bandname]["mean"] = merged[bandname]["sum"] / merged[bandname]["count"]
                merged[bandname]["std"] = np.sqrt((merged[bandname]["sumsq"] / merged[bandname]["count"]) - merged[bandname]["mean"] ** 2)
    return merged

def process_directory_wrapper(args):
    try:
        return process_directory(*args)
    except Exception as e:
        print(f"Error in directory {args[0]}: {e}")

def slow_func(x):
    time.sleep(0.5)
    return x * x

def getdirs(root_dir):
    dirs=[]
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if any(f.endswith(".TIF") for f in filenames):
            dirs.append(dirpath)
    return dirs

def compute_min_max_seq(dirs):
    results=[]
    i=0
    stopc = 3
    for dirpath in dirs:
        if i>=stopc:
            break
        results.append(process_directory(dirpath))
        i+=1
    return results

def normalize_array(image: np.ndarray, band_stats_list: list, normalization_type: str) -> np.ndarray:
    """
    Normalize a 3D image array using provided band stats.

    Parameters:
        image (np.ndarray): 3D array (bands, height, width)
        band_stats_list (list of dict): Each dict must contain 'min', 'max', 'mean', 'std' for each band.
        normalization_type (str): 'minmax', 'zscore', or 'std'

    Returns:
        np.ndarray: Normalized array
    """
    assert image.ndim == 3, "Input image must be a 3D array (bands, height, width)"
    assert len(band_stats_list) == image.shape[0], "Number of band stats must match number of image bands"

    normalized = np.empty_like(image, dtype=np.float32)

    for i, stats in enumerate(band_stats_list):
        band_data = image[i]
        if stats is None:
            normalized[i] = band_data
            continue
        if normalization_type == "minmax":
            normalized[i] = (band_data - stats["min"]) / (stats["max"] - stats["min"])
        elif normalization_type == "zscore":
            normalized[i] = (band_data - stats["mean"]) / stats["std"]
        elif normalization_type == "std":
            normalized[i] = band_data / stats["std"]
        elif normalization_type == "clip_pad_minmax":
            normalized[i] = (band_data - stats["min"]) / (stats["max"] - stats["min"])
        elif normalization_type == "clip_minmax":
            normalized[i] = (band_data - stats["min"]) / (stats["max"] - stats["min"])    
        else:
            raise ValueError(f"Unsupported normalization type '{normalization_type}'. Use 'minmax', 'clip_pad_minmax', 'clip_minmax', 'zscore', or 'std'.")

    return normalized

def match_band_stats(stats_df: pd.DataFrame, band_names: list, exclude_bands: list=[]) -> list:
    """
    Match each band name to a row in stats_df using substring matching.
    
    Parameters:
        stats_df (pd.DataFrame): DataFrame with stats columns incl. 'Band Description', 'min', 'max', 'mean', 'std'.
        band_names (list): List of band name strings.

    Returns:
        list of dict: Each dict contains stats for one band.
    """
    matched_stats = []

    for band_name in band_names:
        # Substring match: Band Description must be in the band_name
        # match = stats_df[stats_df["Band Description"].apply(lambda desc: str(desc).lower() in band_name.lower())]

        # Exact match:
        match = stats_df[stats_df["Band Description"]==band_name]

        if match.empty or band_name in exclude_bands:
            matched_stats.append(None)
            continue
        
        band_stats = match.iloc[0]  # Take the first match
        stats_dict = {
            "band_name": band_name,
            "min": band_stats["min"],
            "max": band_stats["max"],
            "mean": band_stats["mean"],
            "std": band_stats["std"]
        }
        matched_stats.append(stats_dict)
    
    return matched_stats

'''
dirs=getdirs(os.path.expanduser("~/shared_storage/tcloudDS/ROIs_v3_resampled"))
debug=True

start_time = time.time()
#digest = TDigest()

#results = run_in_parallel(process_directory, dirs, max_workers=7, show_progress=True, skipnone=True)
results = compute_min_max_seq(dirs)

end_time = time.time()
elapsed = end_time - start_time

print(f"Execution time: {elapsed:.4f} seconds")

era5_all = [r[0] for r in results]
clear_cloudy_all = [r[1] for r in results]

alldict={"era5_acqtime": merge_dicts(era5_all), "clear_cloudy": merge_dicts(clear_cloudy_all)}

save_min_max_to_csv(alldict,os.path.expanduser('~/nvme1/usgslandsat/DSv3/featurestats.csv'))

'''