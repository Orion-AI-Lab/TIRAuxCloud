import os
import rasterio
from rasterio.mask import mask
import numpy as np
from collections import defaultdict
from collections import Counter
import re
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import nearest_points
from shapely.geometry import box, mapping
import matplotlib.pyplot as plt
import xarray as xr
from pyproj import Transformer
from scipy.ndimage import distance_transform_edt
from datetime import datetime
import geopandas as gpd
from rasterio.warp import calculate_default_transform, reproject, Resampling

def extract_month_from_filename(filename):
    """
    Extracts the acquisition month from the first YYYYMMDD date in the Landsat filename.
    Example: LC09_L1TP_232064_20231205_... → 12
    """
    match = re.search(r'_(\d{8})_', filename)  # Look for 8-digit date in underscores
    if match:
        return int(match.group(1)[4:6])  # Extract MM from YYYYMMDD
    return None

def analyze_patches_by_month(patch_folder):
    """
    Analyzes all .tif patches in a folder and prints how many belong to each month.
    """
    month_counts = defaultdict(int)
    patch_files = [f for f in os.listdir(patch_folder) if f.endswith(".tif")]

    if not patch_files:
        print("No patch files found in the directory.")
        return

    total_patches = len(patch_files)

    for idx, patch_file in enumerate(patch_files, start=1):
        month = extract_month_from_filename(patch_file)
        if month:
            month_counts[month] += 1

        if idx % 10 == 0 or idx == total_patches:
            print(f"Processed {idx}/{total_patches} patches for month analysis...")

    print("\nPatch Distribution by Acquisition Month:")
    months = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    for month_num in range(1, 13):
        print(f"{months[month_num - 1]}: {month_counts[month_num]} patches")

def extract_landsat_version(filename):
    """
    Extracts whether the patch belongs to Landsat 8 or 9 based on the filename prefix.
    Example: LC08_... → Landsat 8
    """
    match = re.match(r'LC(08|09)_', filename)
    if match:
        return match.group(1)  # Returns '08' or '09'
    return None

def count_landsat_versions(patch_folder):
    """
    Counts the number of patches from Landsat 8 and Landsat 9.
    """
    landsat_counts = {"Landsat 8": 0, "Landsat 9": 0}
    patch_files = [f for f in os.listdir(patch_folder) if f.endswith(".tif")]

    if not patch_files:
        print("No patch files found in the directory.")
        return

    total_patches = len(patch_files)

    for idx, patch_file in enumerate(patch_files, start=1):
        version = extract_landsat_version(patch_file)
        if version == "08":
            landsat_counts["Landsat 8"] += 1
        elif version == "09":
            landsat_counts["Landsat 9"] += 1

        if idx % 10 == 0 or idx == total_patches:
            print(f"Processed {idx}/{total_patches} patches for Landsat version analysis...")

    # Print summary
    print("\nLandsat Patch Distribution:")
    print(f"Landsat 8: {landsat_counts['Landsat 8']} patches")
    print(f"Landsat 9: {landsat_counts['Landsat 9']} patches")


def load_etopo_data():    
    file_path = "/home/nvme1/usgslandsat/etopo/ETOPO2v2c_f4.nc"
    
    def coordsdataset_z(maxeast=180, minwest=-180, maxnorth=90, minsouth=-90):
        grid = xr.open_dataset(file_path, engine="netcdf4")
        grid = grid.sel(y=slice(minsouth, maxnorth), x=slice(minwest, maxeast))
        lon2d, lat2d = xr.broadcast(grid.x, grid.y)
        grid = grid.assign(lon=lon2d, lat=lat2d)
        return grid
    
    return coordsdataset_z()


def get_patch_altitude(patch_path, etopo_grid, stat = 'mean'):
    """Extracts the altitude (mean or max) for a patch from ETOPO2."""
    with rasterio.open(patch_path) as patch:
        patch_bounds = patch.bounds
        transformer = Transformer.from_crs(patch.crs, "EPSG:4326", always_xy=True)
        left, bottom = transformer.transform(patch_bounds.left, patch_bounds.bottom)
        right, top = transformer.transform(patch_bounds.right, patch_bounds.top)

        # Add small margin to avoid edge cropping
        lat_margin, lon_margin = 0.01, 0.01
        ymin, ymax = min(bottom, top) - lat_margin, max(bottom, top) + lat_margin
        xmin, xmax = min(left, right) - lon_margin, max(left, right) + lon_margin

        grid_subset = etopo_grid.sel(y=slice(ymin, ymax), x=slice(xmin, xmax))
        elevation_data = grid_subset.z.values

        if elevation_data.size == 0:
            return None
        
        if stat == "max":
            return np.max(elevation_data)
        elif stat == "mean":
            return np.mean(elevation_data)
        else:
            raise ValueError("Invalid stat. Choose 'max' or 'mean'.")

def categorize_altitude(altitude):
    if altitude < 200:
        return "Lowlands (<200m)"
    elif 200 <= altitude < 500:
        return "Uplands (200-500m)"
    elif 500 <= altitude < 1500:
        return "Low Mountains (500-1500m)"
    elif 1500 <= altitude < 2500:
        return "Mid Mountains (1500-2500m)"
    elif 2500 <= altitude < 3500:
        return "High Mountains (2500-3500m)"
    else:
        return "Very High Mountains (>3500m)"

def analyze_patch_altitudes(patch_folder, etopo_grid):
    altitude_categories = defaultdict(int)
    patch_files = [f for f in os.listdir(patch_folder) if f.endswith(".tif")]

    if not patch_files:
        print("No patch files found in the directory.")
        return

    total_patches = len(patch_files)

    for idx, patch_file in enumerate(patch_files, start=1):
        patch_path = os.path.join(patch_folder, patch_file)
        max_altitude = get_patch_altitude(patch_path, etopo_grid)

        if max_altitude is not None:
            category = categorize_altitude(max_altitude)
            altitude_categories[category] += 1

        if idx % 10 == 0 or idx == total_patches:
            print(f"Processed {idx}/{total_patches} patches for altitude analysis...")

    if not altitude_categories:
        print("No altitude data found for the patches.")
        return

    print("\nAltitude Category Distribution:")
    for category, count in altitude_categories.items():
        print(f"{category}: {count} patches")

def init_continents_polygons():
    continents = gpd.read_file("/home/nvme1/usgslandsat/World_Continents/World_Continents.shp")
    print("Loaded continent shapefile.")
    return continents

def get_patch_bounds(patch_path, target_crs="EPSG:3857"):
    with rasterio.open(patch_path) as patch:
        bounds = patch.bounds
        transformer = Transformer.from_crs(patch.crs, target_crs, always_xy=True)
        minx, miny = transformer.transform(bounds.left, bounds.bottom)
        maxx, maxy = transformer.transform(bounds.right, bounds.top)
    return minx, miny, maxx, maxy

def check_continent_from_bounds(bounds, continents):
    minx, miny, maxx, maxy = bounds
    center_point = Point((minx + maxx) / 2, (miny + maxy) / 2)

    for _, row in continents.iterrows():
        if row['geometry'].contains(center_point):
            return row['CONTINENT']

    return find_nearest_continent(center_point, continents)

def find_nearest_continent(point, continents):
    min_distance = float("inf")
    nearest = None
    for _, row in continents.iterrows():
        nearest_pt = nearest_points(point, row['geometry'])[1]
        dist = point.distance(nearest_pt)
        if dist < min_distance:
            min_distance = dist
            nearest = row['CONTINENT']
    return nearest

def assign_continents_to_patches(patch_folder, continents):
    continent_counts = Counter()
    patch_files = [f for f in os.listdir(patch_folder) if f.endswith(".tif")]
    total = len(patch_files)

    for idx, patch_file in enumerate(patch_files, start=1):
        patch_path = os.path.join(patch_folder, patch_file)
        bounds = get_patch_bounds(patch_path, target_crs="EPSG:3857")
        continent = check_continent_from_bounds(bounds, continents)
        continent_counts[continent] += 1

        if idx % 10 == 0 or idx == total:
            print(f"Processed {idx}/{total} patches...")

    return continent_counts

def ContinentsShare():
    shapefile_path = "/home/nvme1/usgslandsat/World_Continents/World_Continents.shp"
    world_continents = gpd.read_file(shapefile_path)
    world_continents = world_continents.to_crs(epsg=6933)  # Equal-area projection
    world_continents["area_km2"] = world_continents.geometry.area / 1e6  # Convert m² to km²
    total_land_area = world_continents["area_km2"].sum()
    world_continents["% of Total Land"] = (world_continents["area_km2"] / total_land_area) * 100

    verified_land_df = world_continents[["CONTINENT", "% of Total Land"]]

    print("\nVerified Land Share (Total Land = 100%)")
    print(verified_land_df.to_string(index=False))  # Print without index

def load_climate_raster(climate_tif_path):    
    with rasterio.open(climate_tif_path) as climate_src:
        return climate_src.crs  

def get_closest_nonzero_class(patch_values):    
    unique, counts = np.unique(patch_values[patch_values != 0], return_counts=True)
    if unique.size > 0:
        return unique[np.argmax(counts)]
    return 0  # Fallback

def process_patches_climate(patches_folder, climate_tif_path):    
    climate_crs = load_climate_raster(climate_tif_path)
    climate_counts = {}

    patch_files = [f for f in os.listdir(patches_folder) if f.endswith(".tif")]
    total_patches = len(patch_files)

    for idx, patch_file in enumerate(patch_files, start=1):
        patch_path = os.path.join(patches_folder, patch_file)

        with rasterio.open(patch_path) as patch_src:
            patch_bounds = patch_src.bounds
            patch_crs = patch_src.crs

            # Reproject bounds if CRS differs
            if patch_crs != climate_crs:
                transformer = Transformer.from_crs(patch_crs, climate_crs, always_xy=True)
                minx, miny = transformer.transform(patch_bounds.left, patch_bounds.bottom)
                maxx, maxy = transformer.transform(patch_bounds.right, patch_bounds.top)
            else:
                minx, miny, maxx, maxy = patch_bounds.left, patch_bounds.bottom, patch_bounds.right, patch_bounds.top

            patch_polygon = [mapping(box(minx, miny, maxx, maxy))]

            with rasterio.open(climate_tif_path) as climate_src:
                out_image, _ = mask(climate_src, patch_polygon, crop=True)
                out_image = out_image[0]  # single band

                unique, counts = np.unique(out_image, return_counts=True)

                if unique.size > 0:
                    dominant_class = unique[np.argmax(counts)]

                    if dominant_class == 0:
                        dominant_class = get_closest_nonzero_class(out_image)

                    climate_counts[dominant_class] = climate_counts.get(dominant_class, 0) + 1

        if idx % 10 == 0 or idx == total_patches:
            print(f"Processed {idx}/{total_patches} patches...")

    return climate_counts

def print_climate_statistics(climate_counts):    
    print("\nKöppen Climate Class Statistics:")
    if climate_counts:
        for climate_class, count in sorted(climate_counts.items()):
            print(f"Climate Class {climate_class}: {count} patches")
    else:
        print("No valid patches found.")  

def ClimateClassesShare():
    raster_path = "/home/shared_storage/ororaDS/landsatDS/koppen_geiger_0p00833333.tif"
    reprojected_path = "/home/shared_storage/ororaDS/landsatDS/koppen_geiger_reprojected.tif"
    
    with rasterio.open(raster_path) as src:
        target_crs = "EPSG:6933"  # Equal-area projection
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        reprojected_meta = src.meta.copy()
        reprojected_meta.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(reprojected_path, "w", **reprojected_meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.nearest
            )

    print("Raster successfully reprojected to EPSG:6933.")

    with rasterio.open(reprojected_path) as src:
        raster_data = src.read(1)
        nodata_value = src.nodata

        pixel_area_km2 = abs(src.transform.a * src.transform.e) / 1e6  # m² to km²

    valid_pixels = raster_data[raster_data != nodata_value]
    unique_classes, counts = np.unique(valid_pixels, return_counts=True)

    total_land_area_km2 = counts.sum() * pixel_area_km2
    class_areas_km2 = counts * pixel_area_km2
    class_percentages = (class_areas_km2 / total_land_area_km2) * 100

    class_df = pd.DataFrame({
        "Class": unique_classes,
        "Area (km²)": class_areas_km2,
        "Coverage (%)": class_percentages
    })

    print("\nKöppen-Geiger Climate Class Distribution (Global):")
    print(class_df.to_string(index=False))              

def calculate_cloud_coverage_band(tif_path, band_name):
    """
    Calculates cloud + thin cloud % for a specific band (clear or cloudy).
    Assumes: 1 = clear, 2 = thin cloud, 3 = cloud
    """
    with rasterio.open(tif_path) as src:
        band_names = src.descriptions
        try:
            idx = band_names.index(band_name) + 1  # 1-based index
        except ValueError:
            print(f"[WARNING] Band {band_name} not found in {tif_path}")
            return None

        cloud_mask = src.read(idx)
        total_pixels = cloud_mask.size

        # Count cloudy + thin cloud pixels (2 and 3)
        cloudy_pixels = np.sum((cloud_mask == 2) | (cloud_mask == 3))

        return (cloudy_pixels / total_pixels) * 100

    
def analyze_patches_cc(patch_folder):
    """
    Analyzes cloud coverage using both clear_cloud_mask and cloudy_cloud_mask.
    Bins coverage into: 0%, 2.5% steps, 100%
    """
    patch_files = [f for f in os.listdir(patch_folder) if f.endswith(".tif")]
    if not patch_files:
        print("No patch files found.")
        return

    band_types = ["clear_cloud_mask", "cloudy_cloud_mask"]
    step = 2.5
    bin_ranges = [(0, 0)] + [(b, b + step) for b in np.arange(0, 102.5, step) if b + step < 102.5] + [(100, 100)]
    bin_labels = (
        ["Exactly 0%"] +
        [f"{start:.1f}–{end:.1f}%" for (start, end) in bin_ranges[1:-1]] +
        ["Exactly 100%", "Unbinned"]
    )
    label_to_index = {label: i for i, label in enumerate(bin_labels)}

    for band in band_types:
        print(f"\n=== {band} ===")
        bin_counts = dict.fromkeys(bin_labels, 0)
        results = []

        for idx, patch_file in enumerate(patch_files, start=1):
            patch_path = os.path.join(patch_folder, patch_file)
            coverage = calculate_cloud_coverage_band(patch_path, band)

            if coverage is None:
                continue

            for (start, end), label in zip(bin_ranges, bin_labels):
                if start == end:
                    if np.isclose(coverage, start, atol=1e-3):
                        bin_label = label
                        break
                elif start < coverage < end:
                    bin_label = label
                    break
            else:
                if coverage >= 99.999:
                    bin_label = "Exactly 100%"
                else:
                    bin_label = "Unbinned"

            bin_counts[bin_label] += 1
            results.append((patch_file, coverage, bin_label))

            if idx % 10 == 0 or idx == len(patch_files):
                print(f"Processed {idx}/{len(patch_files)} patches...")

        print("\nCloud Coverage Statistics:")
        for label in bin_labels:
            print(f"{label}: {bin_counts[label]} patches")

        # Plot histogram
        hist_data = [label_to_index[r[2]] for r in results]
        plt.figure(figsize=(14, 6))
        plt.hist(hist_data, bins=np.arange(-0.5, len(bin_labels), 1), edgecolor='black', rwidth=0.9)
        plt.xticks(ticks=np.arange(len(bin_labels)), labels=bin_labels, rotation=90)
        plt.xlabel("Cloud Coverage (%)")
        plt.ylabel("Number of Patches")
        plt.title(f"Histogram of Cloud Coverage for {band}")
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(f"/home/nvme1/usgslandsat/cc_histogram_{band}.png", dpi=300)
        plt.show()


# === RUNNING POINT ===
if __name__ == "__main__":
    patch_folder = "/home/shared_storage/tcloudDS/patches_v3MA_0717/m100_p256/allnorm"
    analyze_patches_by_month(patch_folder)
    #count_landsat_versions(patch_folder)
    #etopo_grid = load_etopo_data()
    #analyze_patch_altitudes(patch_folder, etopo_grid)
    # continents = init_continents_polygons()
    # continent_counts = assign_continents_to_patches(patch_folder, continents)

    # print("\nPatch Distribution by Continent:")
    # for continent, count in continent_counts.items():
    #     print(f"{continent}: {count} patches")

    
    # ContinentsShare()  

    # climate_tif_path = "/home/shared_storage/ororaDS/landsatDS/koppen_geiger_0p00833333.tif"
    # climate_counts = process_patches_climate(patch_folder, climate_tif_path)
    # print_climate_statistics(climate_counts)

    # ClimateClassesShare()

    # analyze_patches_cc(patch_folder)
