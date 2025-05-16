import pandas as pd
from dataset import *
import os
import glob
import numpy as np
import rasterio
import matplotlib.pyplot as plt

def get_raster_value_distribution(directories, bins=100):
    """
    Compute the distribution of biomass values from .tif raster images in the given directories.

    Parameters:
    - directories: List of paths to search for .tif files.
    - bins: Number of bins to use for the histogram.

    Returns:
    - hist: Histogram values.
    - bin_edges: Edges of the bins.
    - total_pixels: Total number of valid pixels processed.
    - value_array: Concatenated array of all valid pixel values (for advanced stats).
    """
    value_array = []

    for directory in directories:
        tif_files = glob.glob(os.path.join(directory, '**', '*.tif'), recursive=True)

        for tif_path in tif_files:
            try:
                with rasterio.open(tif_path) as src:
                    data = src.read(1)  # Read the first band
                    nodata = src.nodata

                    # Flatten and filter valid values
                    data_flat = data.flatten()
                    if nodata is not None:
                        data_flat = data_flat[data_flat != nodata]
                    data_flat = data_flat[~np.isnan(data_flat)]

                    value_array.append(data_flat)
            except Exception as e:
                print(f"Failed to process {tif_path}: {e}")

    # Combine all values
    all_values = np.concatenate(value_array)
    hist, bin_edges = np.histogram(all_values, bins=bins)

    return hist, bin_edges, len(all_values), all_values

def plot_raster_distribution(hist, bin_edges, title="Biomass Value Distribution"):
    """
    Plot histogram of raster values.

    Parameters:
    - hist: Histogram values.
    - bin_edges: Bin edges for the histogram.
    - title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], hist, width=np.diff(bin_edges), align='edge', edgecolor='black')
    plt.xlabel("Biomass Value")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('Distribution_of_data.png')
    # plt.show()

def save_img(img_array):
    norm_img = (255 * (img_array - img_array.min()) / (img_array.max() - img_array.min())).astype(np.uint8)
    plt.imsave('SAR_image.png', norm_img)



if __name__ == "__main__":
    # df = pd.read_csv("data/features_metadata.csv")
    # test_df = df[df.split == "test"].copy()
    # test_df = test_df.groupby("chip_id").agg(list).reset_index()
    # print(test_df)

    # test_images_dir = 'data/test_features'
    # test_dataset = DS(
    #     df=test_df,
    #     dir_features=test_images_dir,
    #     veg_indices=True
    # )

    # for idx in range(len(test_dataset)):
    #     img, mask, target = test_dataset[idx]
    #     print(f"Data at index {idx}:", img.shape)
    #     sar_img = img[0][11:15].reshape((256,256,4))
    #     break

    directories = ["data/train_agbm", "data/test_agbm"]
    hist, bin_edges, total_pixels, all_values = get_raster_value_distribution(directories)

    print(f"Processed {total_pixels} valid pixels from all rasters.")
    print(f"Min value: {np.min(all_values):.2f}, Max value: {np.max(all_values):.2f}, Mean: {np.mean(all_values):.2f}")
    
    plot_raster_distribution(hist, bin_edges)

    pass