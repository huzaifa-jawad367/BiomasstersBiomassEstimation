import os
import numpy as np
import rasterio
import pandas as pd
from collections import defaultdict

# import xarray as xr
# from xrspatial import multispectral

# def calculate_veg_indices_uint8(img_s2):
#     """
#     Calculate vegetation indices and convert them to uint8.
#     Args:
#         img_s2 (np.ndarray): Sentinel-2 image array (H, W, Bands)
    
#     Returns:
#         dict: A dictionary of vegetation indices scaled to uint8
    
#     # Calculate vegetation indices
#     ndvi = (img_s2[:, :, 6] - img_s2[:, :, 2]) / (img_s2[:, :, 6] + img_s2[:, :, 2] + epsilon)
#     evi = (2.5 * (img_s2[:, :, 6] - img_s2[:, :, 2])) / (
#         img_s2[:, :, 6] + 6 * img_s2[:, :, 2] - 7.5 * img_s2[:, :, 0] + 1 + epsilon
#     )
#     savi = ((img_s2[:, :, 6] - img_s2[:, :, 2]) * (1 + 0.5)) / (img_s2[:, :, 6] + img_s2[:, :, 2] + 0.5 + epsilon)
#     msavi = 0.5 * (
#         2 * img_s2[:, :, 6] + 1 - np.sqrt(
#             np.square(2 * img_s2[:, :, 6] + 1) - 8 * (img_s2[:, :, 6] - img_s2[:, :, 2]) + epsilon
#         )
#     )
#     ndmi = (img_s2[:, :, 6] - img_s2[:, :, 7]) / (img_s2[:, :, 6] + img_s2[:, :, 7] + epsilon)
#     nbr = (img_s2[:, :, 6] - img_s2[:, :, 8]) / (img_s2[:, :, 6] + img_s2[:, :, 8] + epsilon)
#     nbr2 = (img_s2[:, :, 7] - img_s2[:, :, 8]) / (img_s2[:, :, 7] + img_s2[:, :, 8] + epsilon)

#     """
#     img_s2_xr = xr.DataArray(img_s2)

#     # Define epsilon to avoid division by zero
#     epsilon = 1e-6
    
#     # NDVI - Normalized Difference Vegetation Index
#     ndvi = np.array(multispectral.ndvi(img_s2_xr[:, :, 6], img_s2_xr[:, :, 2]))
    
#     # EVI - Enhanced Vegetation Index
#     evi = np.array(multispectral.evi(img_s2_xr[:, :, 6], img_s2_xr[:, :, 2], img_s2_xr[:, :, 0]))
    
#     # SAVI - Soil-Adjusted Vegetation Index
#     savi = np.array(multispectral.savi(img_s2_xr[:, :, 6], img_s2_xr[:, :, 2]))
    
#     # # MSAVI - Modified Soil-Adjusted Vegetation Index
#     # msavi = multispectral.msavi(img_s2_xr[:, :, 6], img_s2_xr[:, :, 2])
#     msavi = 0.5 * (
#         2 * img_s2[:, :, 6] + 1 - np.sqrt(
#             np.square(2 * img_s2[:, :, 6] + 1) - 8 * (img_s2[:, :, 6] - img_s2[:, :, 2])
#         )
#     )
    
#     # NDMI - Normalized Difference Moisture Index
#     ndmi = np.array(multispectral.ndmi(img_s2_xr[:, :, 6], img_s2_xr[:, :, 7]))
    
#     # NBR - Normalized Burn Ratio
#     nbr = np.array(multispectral.nbr(img_s2_xr[:, :, 6], img_s2_xr[:, :, 8]))
    
#     # NBR2 - Another variation of Normalized Burn Ratio
#     nbr2 = np.array(multispectral.nbr2(img_s2_xr[:, :, 7], img_s2_xr[:, :, 8]))

#     # Normalize indices to [0, 255] and convert to uint8
#     def normalize_and_convert(index):
#         index_normalized = (index + 1) / 2  # Scale [-1, 1] to [0, 1]
#         # index_uint8 = (index_normalized * 255).clip(0, 255).astype("uint8")  # Scale to [0, 255] and convert to uint8
#         return index_normalized

#     # Create a dictionary of uint8 vegetation indices
#     indices_uint8 = {
#         "ndvi": normalize_and_convert(ndvi),
#         "evi": normalize_and_convert(evi),
#         "savi": normalize_and_convert(savi),
#         "msavi": normalize_and_convert(msavi),
#         "ndmi": normalize_and_convert(ndmi),
#         "nbr": normalize_and_convert(nbr),
#         "nbr2": normalize_and_convert(nbr2),
#     }

#     return indices_uint8

# # Path to the directory containing the images
# directory_path = "data/Volumes/Samsung_T5/BIOMASS/BioMasster_Dataset/v1/DrivenData/train_features"

# # Read metadata and get a list of filenames for Sentinel-2
# data_files = pd.read_csv('data/features_metadata.csv')
# filenames = list(data_files[data_files['satellite'] == 'S2']['filename'])

# # Dictionary to accumulate all pixel values for each vegetation index
# # Instead of storing just max, we'll collect all values to compute mean, median, std as well.
# stats_arrays = {
#     "ndvi": [],
#     "evi": [],
#     "savi": [],
#     "msavi": [],
#     "ndmi": [],
#     "nbr": [],
#     "nbr2": []
# }

# # Initialize dictionaries to store statistics
# all_stats = defaultdict(lambda: {"max": None, "mean": None, "median": None, "std": None})
# file_count = defaultdict(int)  # Count of valid files processed for each index

# # Process each image to compute vegetation indices and accumulate statistics
# for filename in filenames:
#     file_path = os.path.join(directory_path, filename)

#     # Process only valid image files (assuming .tif format)
#     if os.path.isfile(file_path) and filename.endswith(".tif"):
#         try:
#             with rasterio.open(file_path) as src:
#                 # Read image data (Bands, Height, Width)
#                 img = src.read()
#                 # Move axis to have shape (Height, Width, Bands)
#                 img = np.moveaxis(img, 0, -1)

#                 # Calculate vegetation indices
#                 indices_uint8 = calculate_veg_indices_uint8(img)

#                 # Update statistics for each index
#                 for index_name, index_values in indices_uint8.items():
#                     # Ensure index_values is a NumPy array
#                     index_values_np = index_values.values if hasattr(index_values, "values") else index_values

#                     # Flatten the array for calculations
#                     index_flat = index_values_np.ravel()

#                     # Update statistics incrementally
#                     if all_stats[index_name]["max"] is None:
#                         # First file initialization
#                         all_stats[index_name]["max"] = np.max(index_flat)
#                         all_stats[index_name]["mean"] = np.mean(index_flat)
#                         all_stats[index_name]["median"] = np.median(index_flat)
#                         all_stats[index_name]["std"] = np.std(index_flat)
#                     else:
#                         # Incremental update for mean and std (approximation)
#                         current_count = file_count[index_name]
#                         new_count = current_count + 1

#                         # Update max
#                         all_stats[index_name]["max"] = max(all_stats[index_name]["max"], np.max(index_flat))
#                         # Weighted mean update
#                         all_stats[index_name]["mean"] = (
#                             (all_stats[index_name]["mean"] * current_count + np.mean(index_flat)) / new_count
#                         )
#                         # Approximation: re-compute median and std from scratch
#                         # Drop median and std if re-computation is infeasible due to memory constraints
#                         try:
#                             combined_data = np.concatenate(
#                                 [np.full(current_count, all_stats[index_name]["median"]), index_flat]
#                             )
#                             all_stats[index_name]["median"] = np.median(combined_data)
#                             all_stats[index_name]["std"] = np.std(combined_data)
#                         except MemoryError:
#                             all_stats[index_name]["median"] = None
#                             all_stats[index_name]["std"] = None

#                     file_count[index_name] += 1  # Increment file count for this index

#         except Exception as e:
#             print(f"Error processing file {filename}: {e}")

# # Print the results
# print("Statistics for each vegetation index across processed images:")
# for index_name, stats_dict in all_stats.items():
#     print(f"Index: {index_name}")
#     for stat_name, value in stats_dict.items():
#         if value is not None:
#             print(f"  {stat_name}: {value}")
#         else:
#             print(f"  {stat_name}: Not calculated (dropped due to memory constraints)")
#     print()

# # Path to the .tif file
# file_path = "data/train_agbm/a1e2da4b_agbm.tif"

# # Open the .tif file and read it as an array
# with rasterio.open(file_path) as src:
#     tif_array = src.read()  # Reads all bands as a NumPy array

# # Print the array
# print(len(np.unique(tif_array)))

# # If you want to see the shape of the array
# print("Array Shape:", tif_array.shape)  # (Bands, Height, Width)



# print(len(os.listdir("data/train_agbm")))
# df = pd.read_csv("data/features_metadata.csv")
# df = df[df['split'] == 'train']['corresponding_agbm']
# df_list = df.unique()
# print(len(df_list))