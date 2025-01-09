import os
import numpy as np
import rasterio
import pandas as pd

# Function to calculate vegetation indices (already provided, copy here for use)
def calculate_veg_indices_uint8(img_s2):
    epsilon = 1e-6  # Small value to prevent division by zero
    
    ndvi = (img_s2[:, :, 6] - img_s2[:, :, 2]) / (img_s2[:, :, 6] + img_s2[:, :, 2] + epsilon)
    evi = (2.5 * (img_s2[:, :, 6] - img_s2[:, :, 2])) / (
        img_s2[:, :, 6] + 6 * img_s2[:, :, 2] - 7.5 * img_s2[:, :, 0] + 1 + epsilon
    )
    savi = ((img_s2[:, :, 6] - img_s2[:, :, 2]) * (1 + 0.5)) / (img_s2[:, :, 6] + img_s2[:, :, 2] + 0.5)
    msavi = 0.5 * (
        2 * img_s2[:, :, 6] + 1 - np.sqrt(
            np.square(2 * img_s2[:, :, 6] + 1) - 8 * (img_s2[:, :, 6] - img_s2[:, :, 2])
        )
    )
    ndmi = (img_s2[:, :, 6] - img_s2[:, :, 7]) / (img_s2[:, :, 6] + img_s2[:, :, 7] + epsilon)
    nbr = (img_s2[:, :, 6] - img_s2[:, :, 8]) / (img_s2[:, :, 6] + img_s2[:, :, 8] + epsilon)
    nbr2 = (img_s2[:, :, 7] - img_s2[:, :, 8]) / (img_s2[:, :, 7] + img_s2[:, :, 8] + epsilon)
    
    def normalize_and_convert(index):
        index_normalized = (index + 1) / 2
        index_uint8 = (index_normalized * 255).clip(0, 255).astype("uint8")
        return index_uint8

    indices_uint8 = {
        # "ndvi": normalize_and_convert(ndvi),
        # "evi": normalize_and_convert(evi),
        # "savi": normalize_and_convert(savi),
        # "msavi": normalize_and_convert(msavi),
        # "ndmi": normalize_and_convert(ndmi),
        # "nbr": normalize_and_convert(nbr),
        # "nbr2": normalize_and_convert(nbr2),
        "ndvi": ndvi,
        "evi": evi,
        "savi": savi,
        "msavi": msavi,
        "ndmi": ndmi,
        "nbr": nbr,
        "nbr2": nbr2,
    }

    return indices_uint8

# Path to the directory containing the images
directory_path = "data/Volumes/Samsung_T5/BIOMASS/BioMasster_Dataset/v1/DrivenData/train_features"

# Initialize max values for each vegetation index
max_indices = {
    "ndvi": 0,
    "evi": 0,
    "savi": 0,
    "msavi": 0,
    "ndmi": 0,
    "nbr": 0,
    "nbr2": 0,
}

data_files = pd.read_csv('data/features_metadata.csv')
filenames = list(data_files[data_files['satellite'] == 'S2']['filename'])

# Process each image to compute vegetation indices and update max values
for filename in filenames:
    file_path = os.path.join(directory_path, filename)

    # Process only valid image files (assuming .tif format)
    if os.path.isfile(file_path) and filename.endswith(".tif"):
        with rasterio.open(file_path) as src:
            img = src.read()  # Shape: (Bands, Height, Width)
            img = np.moveaxis(img, 0, -1)  # Convert to (Height, Width, Bands)

            # Calculate vegetation indices
            indices_uint8 = calculate_veg_indices_uint8(img)

            # Update max values for each index
            for index_name, index_values in indices_uint8.items():
                max_indices[index_name] = max(max_indices[index_name], index_values.max())


# Print maximum values for all vegetation indices
print("Maximum pixel values for vegetation indices:")
for index, max_value in max_indices.items():
    print(f"{index}: {max_value}")
