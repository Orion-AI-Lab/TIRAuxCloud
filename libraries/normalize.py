import numpy as np
import rasterio

def normalize_tif(input_tif_path, bandminmaxdict, minlabel='Min', maxlabel='Max'):
    with rasterio.open(input_tif_path) as src:
        data = src.read().astype(np.float32)
        data_norm = np.empty_like(data)
        for i in range(data.shape[0]):
            band_name = src.descriptions[i]
            band = data[i]
            # band_min, band_max = band.min(), band.max()
            if bandminmaxdict[band_name][maxlabel] > bandminmaxdict[band_name][minlabel]:
                minv = bandminmaxdict[band_name][minlabel]
                maxv = bandminmaxdict[band_name][maxlabel]
                data_norm[i] = (band - minv) / (maxv - minv)
            else:
                data_norm[i] = band  # skip normalization if constant
    return data_norm

