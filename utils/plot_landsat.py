# PATCH_PATH = "/home/shared_storage/tcloudDS/patches_v3_full/m100_p256/all/LC09_L1TP_162019_20230704_20230704_02_T1_592185_6496815_all_100m_256p.tif" 
# SAVE_DIR = "/home/nvme1/usgslandsat/DSv3"

import os
import sys
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# ==== USER CONFIGURATION ====
PATCH_PATH = "/home/shared_storage/tcloudDS/patches_v3_full/m100_p256/allnorm/LC09_L1TP_162019_20230704_20230704_02_T1_592185_6496815_allnorm_100m_256p_norm.tif"
SAVE_DIR = "/home/nvme1/usgslandsat/DSv3"

BANDS_TO_PLOT = [
    'cloudy_Radiance_B10_B11_mean',
    'cloudy_cloud_mask',
    'clear_Radiance_B10_B11_mean',
    'dem',
    'lulc',
    'acqt_surface_pressure',
    'acqt_skin_temp'
]

MAP_FEATURES = {
    'cloudy_Radiance_B10_B11_mean': 'Cloudy Radiance B10/B11',
    'cloudy_cloud_mask': 'Cloudy Cloud Mask',
    'clear_Radiance_B10_B11_mean': 'Clear Radiance B10/B11',
    'dem': 'DEM',
    'lulc': 'LULC',
    'acqt_surface_pressure': 'Surface Pressure',
    'acqt_skin_temp': 'Skin Temperature',
    'rgb_cloudy': 'Cloudy RGB',
    'rgb_clear': 'Clear RGB'
}
# =============================

def get_colormap(desc):
    return 'gray'

def normalize01(x):
    x = x.astype(np.float32)
    return np.clip((x - np.min(x)) / (np.max(x) - np.min(x) + 1e-6), 0, 1)

def normalize_percentile(x, lower=2, upper=98):
    x = x.astype(np.float32)
    vmin = np.percentile(x, lower)
    vmax = np.percentile(x, upper)
    return np.clip((x - vmin) / (vmax - vmin + 1e-6), 0, 1)

def plot_patch(patch_path, bands_to_plot, map_features, save_dir):
    with rasterio.open(patch_path) as src:
        band_descriptions = src.descriptions
        print("Available bands:\n", band_descriptions)

        # Collect valid band indices
        plot_items = []
        for b in bands_to_plot:
            if b in band_descriptions:
                idx = band_descriptions.index(b)
                plot_items.append((idx, b))
            else:
                print(f"[WARNING] Band '{b}' not found in file.")

        # Locate RGB band indices
        try:
            c_b4_idx = band_descriptions.index('cloudy_B4')
            c_b3_idx = band_descriptions.index('cloudy_B3')
            c_b2_idx = band_descriptions.index('cloudy_B2')
            has_cloudy_rgb = True
        except ValueError:
            print("[WARNING] Could not find cloudy_B4, B3, B2.")
            has_cloudy_rgb = False

        try:
            clr_b4_idx = band_descriptions.index('clear_B4')
            clr_b3_idx = band_descriptions.index('clear_B3')
            clr_b2_idx = band_descriptions.index('clear_B2')
            has_clear_rgb = True
        except ValueError:
            print("[WARNING] Could not find clear_B4, B3, B2.")
            has_clear_rgb = False

        # Insert RGB composites
        if has_cloudy_rgb:
            try:
                idx = next(i + 1 for i, (_, b) in enumerate(plot_items) if b == 'cloudy_Radiance_B10_B11_mean')
            except StopIteration:
                idx = 1
            plot_items.insert(idx, (-1, 'rgb_cloudy'))

        if has_clear_rgb:
            try:
                idx = next(i + 1 for i, (_, b) in enumerate(plot_items) if b == 'clear_Radiance_B10_B11_mean')
            except StopIteration:
                idx = len(plot_items)
            plot_items.insert(idx, (-2, 'rgb_clear'))

        # Plotting
        fig, axs = plt.subplots(1, len(plot_items), figsize=(4 * len(plot_items), 4))
        if len(plot_items) == 1:
            axs = [axs]

        for ax, (idx, desc) in zip(axs, plot_items):
            if desc == 'rgb_cloudy' and has_cloudy_rgb:
                r = normalize_percentile(src.read(c_b4_idx + 1))
                g = normalize_percentile(src.read(c_b3_idx + 1))
                b = normalize_percentile(src.read(c_b2_idx + 1))
                rgb = np.stack([r, g, b], axis=-1)
                ax.imshow(rgb)

            elif desc == 'rgb_clear' and has_clear_rgb:
                r = normalize01(src.read(clr_b4_idx + 1))
                g = normalize01(src.read(clr_b3_idx + 1))
                b = normalize01(src.read(clr_b2_idx + 1))
                rgb = np.stack([r, g, b], axis=-1)
                ax.imshow(rgb)

            else:
                img = src.read(idx + 1)
                ax.imshow(img, cmap=get_colormap(desc))

            ax.set_title(map_features.get(desc, desc), fontsize=25)
            ax.axis('off')

        plt.tight_layout()

        # Save figure
        base_name = os.path.splitext(os.path.basename(patch_path))[0]
        save_path = os.path.join(save_dir, f"{base_name}_preview.png")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved preview to: {save_path}")

        plt.show()

if __name__ == "__main__":
    plot_patch(PATCH_PATH, BANDS_TO_PLOT, MAP_FEATURES, SAVE_DIR)
