import torch
import numpy as np
import pandas as pd
import numpy as np
import torch
import rasterio
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
#from torchvision import transforms # Import torchvision transforms
import rioxarray as rxr
from tqdm import tqdm
import torchvision.transforms.functional as TF

def check_valid_values_pytorch(tensor, valid_values=[0, 1, 2]):
    """Check if tensor contains only values from valid_values"""
    unique_vals = torch.unique(tensor)
    valid_tensor = torch.tensor(valid_values, dtype=tensor.dtype, device=tensor.device)

    # Method 1: Using isin (PyTorch 1.10+)
    return torch.all(torch.isin(unique_vals, valid_tensor))

def getfiles(df, patches_path=None):
    if patches_path:
        files = [os.path.join(patches_path,f) for f in [os.path.basename(ff) for ff in df["file"].tolist()]]
    else:
        files = [os.path.expanduser(f) for f in df["file"].tolist()]
    return files

def preload_band_maps(files):
    """
    Optimized band mapping using a Smart Cache dictionary.
    Reduces disk I/O by caching metadata for unique sensor types.
    """
    band_maps = [] # Using a list is slightly cleaner than a dict with indices
    sensor_cache = {} 
    for path in tqdm(files, desc="Loading band maps"):
        with rasterio.open(path) as src:
            # Fingerprint: (Band Count, First Band Name)
            fingerprint = (src.count, src.descriptions[0])   
            if fingerprint not in sensor_cache:
                # Map once per unique sensor type
                sensor_cache[fingerprint] = {desc: j for j, desc in enumerate(src.descriptions)}  
            # Append the reference to the cached dict
            band_maps.append(sensor_cache[fingerprint])
    return band_maps

def loadbands(path, band_map, input_bands_lists, target_band):
    ds = rxr.open_rasterio(path)#, chunks=True)

    xlist = []
    if isinstance(input_bands_lists[0],str): 
        # the items in the list are strings and represent the input features
        input_bands_l = [input_bands_lists] 
    elif isinstance(input_bands_lists[0],list) and input_bands_lists[0][0]=="<or>": 
        # the items in the list are lists and represent aternative feature names 
        # the first item value of the first list is "<or>" to indicate the usage of the list
        # useful for joined training of different domains
        input_bands_l = [input_bands_lists[0][1:]]+input_bands_lists[1:]
    elif isinstance(input_bands_lists[0],list): 
        # the items in the list are lists and represent different endoder input lists
        input_bands_l = input_bands_lists       

    # Read only the specific bands you need (1-based indexing for rioxarray)
    for input_bands in input_bands_l:
        input_indices = [band_map[b] for b in input_bands if b in band_map]
        x = ds.isel(band=input_indices).values  # shape: (len(input_bands), H, W)
        xlist.append(x)
    
    if isinstance(target_band,str):
        target_index = band_map[target_band]
    elif isinstance(target_band,list):
        band = next(band for band in band_map if band in target_band)
        target_index = band_map[band]

    y = ds.isel(band=[target_index]).values.squeeze(0)  # shape: (H, W)

    if len(xlist) == 1:
        return x,y
    else:
        return xlist,y

def process_mask(y, yshift=1, thincloudclass=1):
    y = y - yshift  # Shift from 1–3 to 0–2
    if thincloudclass == 2:
        y[y == 2] = 1
    elif thincloudclass == 0:
        y[y == 1] = 0
        y[y == 2] = 1
    return y


class MultiBandTiffDataset(Dataset):

    def __init__(self, df, band_stats, input_bands, target_band, yshift=1, transform=None, thincloudcl=None, patches_path=None):
        self.files = getfiles(df, patches_path)
        self.band_map = preload_band_maps(self.files)
        self.band_stats = band_stats
        self.input_bands = input_bands
        self.target_band = target_band
        self.transform = transform
        self.thincloudclass = thincloudcl
        self.yshift = yshift
        self.patches_path = patches_path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        patch_path = self.files[idx]
        x,y = loadbands(patch_path, self.band_map[idx], self.input_bands, self.target_band)

        y = process_mask(y, self.yshift, self.thincloudclass)

        # DEBUG print
        #print("\n----- BEFORE TRANSFORM PREP -----")
        #print("x type:", type(x), "shape:", getattr(x, "shape", None))
        #print("y type:", type(y), "shape:", getattr(y, "shape", None))

        if self.transform:
            # === Compute class 1+2 ratio ===
            class_mask = (y == 1) | (y == 2)
            class_ratio = class_mask.sum() / y.size

            # === Conditionally apply transform ===
            if 0.3 <= class_ratio <= 0.7 :

                # DEBUG print
                #print("\n>>> JUST BEFORE ALBUMENTATIONS <<<")
                #print("x (image) shape going in:", x.shape)
                #print("y (mask)  shape going in:", y.shape)
                #print("x dtype:", x.dtype, "y dtype:", y.dtype)
                #print("unique mask values:", np.unique(y) if isinstance(y, np.ndarray) else torch.unique(y))
                #print("class ratio:", class_ratio)
                #print("Applying transform...")
                augmented = self.transform(image=x.transpose(1, 2, 0), mask=y)
                #augmented = self.transform(image=x, mask=y)
                x = augmented["image"]#.permute(2, 0, 1)  # CHW
                y = augmented["mask"]

        # Ensure x, y are PyTorch tensors before calling .float(), .long()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        if not check_valid_values_pytorch(y):
            print("Wrong Tensor!!")
            pass

        return x.float(), y.long()
        

def _debug_sample(x, y, path=None):
    import numpy as _np
    def stats(a):
        try:
            return dict(shape=getattr(a, "shape", None),
                        dtype=getattr(a, "dtype", None),
                        min=_np.nanmin(a).item() if _np.size(a) else None,
                        max=_np.nanmax(a).item() if _np.size(a) else None,
                        mean=_np.nanmean(a).item() if _np.size(a) else None,
                        any_nan=_np.isnan(a).any().item() if _np.size(a) else None,
                        size=_np.size(a))
        except Exception as e:
            return {"error": str(e)}
    print("DATASET DEBUG", "path:", path)
    print(" image stats:", stats(x))
    print(" mask  stats:", stats(y))

class Crop224loader(Dataset):

    def __init__(self, df, band_stats, input_bands, target_band, yshift=1, transform=None, thincloudcl=None, patches_path=None):
        self.files = getfiles(df, patches_path)
        self.band_map = preload_band_maps(self.files)
        self.band_stats = band_stats
        self.input_bands = input_bands
        self.target_band = target_band
        self.transform = transform
        self.thincloudclass = thincloudcl
        self.yshift = yshift
        self.patches_path = patches_path

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        patch_path = self.files[idx]
        x,y = loadbands(patch_path, self.band_map[idx], self.input_bands, self.target_band)
    

        y = process_mask(y, self.yshift, self.thincloudclass)

        TARGET = (224, 224)  # (H, W)

        # Ensure x is a tensor CHW and y is tensor HxW (or 1xHxW)
        # (You already convert earlier; double-check)
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
            # If x originally HWC, convert to CHW
            if x.ndim == 3 and x.shape[2] == len(self.input_bands):
                x = x.permute(2, 0, 1)

        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        # --- Now enforce deterministic center crop on both x and y ---
        # x: (C, H, W)
        if x.ndim != 3:
            raise RuntimeError(f"Unexpected x ndim {x.ndim} for path {patch_path}")

        # Prepare mask: make it 1xHxW to use the same TF.center_crop call
        mask_was_2d = False
        if y.ndim == 2:
            y = y.unsqueeze(0)   # (1, H, W)
            mask_was_2d = True
        elif y.ndim == 3 and y.shape[0] != 1:
            # if mask has multiple channels, leave it as-is (but ensure semantics are intended)
            pass

        # Use torchvision.functional.center_crop which works with CHW tensors
        x = TF.center_crop(x, TARGET)   # -> (C, 224, 224)
        y = TF.center_crop(y, TARGET)   # -> (1, 224, 224) or (C_mask, 224, 224)

        # squeeze mask back to HxW if it was 2D originally
        if mask_was_2d:
            y = y.squeeze(0)   # -> (224, 224)

        # Final sanity check: shapes must match
        if x.shape[-2:] != y.shape[-2:]:
            raise RuntimeError(f"After cropping shapes mismatch for {patch_path}: x {x.shape} vs y {y.shape}")        
        return x.float(), y.long()
    

class CloudyClearDataset(Dataset):
    def __init__(self, df, band_stats, cloudy_bands, clear_bands, target_band,
                 include_clear_mask=False, transform=None, thincloudcl=None, dir=None):
        self.files = getfiles(df, patches_path=None)
        self.band_map = preload_band_maps(self.files)
        self.band_stats = band_stats
        self.cloudy_bands = cloudy_bands
        self.clear_bands = clear_bands
        self.target_band = target_band
        self.include_clear_mask = include_clear_mask
        self.transform = transform
        self.thincloudclass = thincloudcl

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        cclist, target = loadbands(path, self.band_map, [self.cloudy_bands, self.clear_bands], self.target_band)

        cloudy = cclist[0]
        clear = cclist[1]

        target = process_mask(target, thincloudclass=self.thincloudclass)
        
        if self.include_clear_mask:
            clear_mask, _ = loadbands(path, self.band_map, ["clear_cloud_mask"], self.target_band)
            #clear_mask = data[band_map["clear_cloud_mask"]]
            clear = np.concatenate([clear, clear_mask[np.newaxis, :, :]], axis=0)

            # Add zero channel to cloudy to keep shape compatible
            zero_channel = np.zeros_like(clear_mask)[np.newaxis, :, :]
            cloudy = np.concatenate([cloudy, zero_channel], axis=0)

         # Optional transform
        if self.transform:
            # Use cloudy bands for augmentation base
            augmented = self.transform(image=cloudy.transpose(1, 2, 0), mask=target)
            target = augmented["mask"]

        # Convert to tensors
        cloudy = torch.from_numpy(cloudy).float()
        clear = torch.from_numpy(clear).float()
        target = torch.from_numpy(target).long()

        return (cloudy, clear), target



def get_loaders(csv_path, band_stats, input_bands, target_band, yshift=1, clear_bands=None, batch_size=4, 
                thincloudcl=1, transformkey=None, model_type="Unet", testrun=False, dataset_dir=None, workers=4):

    df = pd.read_csv(csv_path)

    if not testrun:
        train_df = df[df["dataset"] == "train"]
        testval_df = df[df["dataset"] == "val"]
    else:
        testval_df = df[df["dataset"] == "test"]    

    if transformkey=="full":
        print("#### Applying Augmentation ####")
        transform = A.Compose([
        A.RandomCrop(256, 256),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(),
        ToTensorV2()
        ])
    else:
        transform = None

    basemodels=["Unet","SegFormer","DeepLabV3","HRCloudNet", "CDnetV2"]
    allbasemodels=["Fine Tune "+bm for bm in basemodels]+basemodels
    if model_type in allbasemodels:
        if not testrun:
            train_ds = MultiBandTiffDataset(train_df, band_stats, input_bands, 
                                            target_band=target_band, yshift=yshift, transform=transform, 
                                            thincloudcl=thincloudcl, patches_path=dataset_dir)
        testval_ds = MultiBandTiffDataset(testval_df, band_stats, input_bands,  
                                          target_band=target_band, yshift=yshift, transform=transform, 
                                          thincloudcl=thincloudcl, patches_path=dataset_dir)
    elif model_type in ["BEFUnet", "Swin-Unet", "SwinCloud"]:
        if not testrun:
            train_ds = Crop224loader(train_df, band_stats, input_bands, 
                                     target_band=target_band, yshift=yshift, transform=transform, 
                                     thincloudcl=thincloudcl, patches_path=dataset_dir)
        testval_ds = Crop224loader(testval_df, band_stats, input_bands, 
                                   target_band=target_band, yshift=yshift, transform=transform, 
                                   thincloudcl=thincloudcl, patches_path=dataset_dir)
    elif model_type in ["Siamese", "bam-cd"]:
        if not testrun:
            train_ds = CloudyClearDataset(df=train_df,band_stats=band_stats,cloudy_bands=input_bands,clear_bands=clear_bands,target_band=target_band,
                                include_clear_mask=False,transform=transform,thincloudcl=thincloudcl, patches_path=dataset_dir)
        testval_ds = CloudyClearDataset(df=testval_df,band_stats=band_stats,cloudy_bands=input_bands,clear_bands=clear_bands,target_band=target_band,
                            include_clear_mask=False,transform=transform,thincloudcl=thincloudcl, patches_path=dataset_dir)

    if not testrun:
        DLtrain = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    else:
        DLtrain = None
    DLvaltest = DataLoader(testval_ds, batch_size=batch_size, num_workers=workers, pin_memory=True)

    return DLtrain, DLvaltest