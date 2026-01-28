import rasterio
import numpy as np

"""
fpath = "/home/shared_storage/tcloudDS/patches_v3/m100_p256/allnorm/LC09_L1TP_232064_20231205_20231205_02_T1_592085_-633785_allnorm_100m_256p_norm.tif"
with rasterio.open(fpath) as src:
    print("File:", fpath)
    print("Band count:", src.count)
    print("Band descriptions:", src.descriptions)
    print("Shape:", src.height, src.width)
    print("Dtype:", src.dtypes)
    print("CRS:", src.crs)
    print("Transform:", src.transform)
"""

f = "/home/shared_storage/tcloudDS/patches_v3/m100_p256/allnorm/LC09_L1TP_232064_20231205_20231205_02_T1_592085_-633785_allnorm_100m_256p_norm.tif"
with rasterio.open(f) as src:
    print(src.descriptions)
    mask = src.read(src.descriptions.index("clear_cloud_mask") + 1)
    print(np.unique(mask, return_counts=True))