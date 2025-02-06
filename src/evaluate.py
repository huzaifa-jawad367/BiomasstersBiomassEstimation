import rasterio
from rasterio.transform import from_origin
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

LIST_OF_FILES = os.listdir('data/test_agbm')

# def draw_difference_

def save_image(file_name):
    pred_file = f"pred_50ep/{file_name}"  # Replace with your TIFF file path
    gt_file = f"data/test_agbm/{file_name}"

    # Convert TIFF to PNG
    with Image.open(gt_file) as img:
        raster_gt = np.array(img)
    with Image.open(pred_file) as img:
        raster_pred = np.array(img)

    normalized_data_gt = (255 * (raster_gt - raster_gt.min()) / (raster_gt.max() - raster_gt.min())).astype(np.uint8)
    normalized_data_pred = (255 * (raster_pred - raster_pred.min()) / (raster_pred.max() - raster_pred.min())).astype(np.uint8)


    colormap = plt.cm.viridis  # You can change this to other colormaps like 'plasma', 'inferno', etc.
    colored_image_gt = colormap(normalized_data_gt / 255.0)  # Normalize to 0-1 for colormap
    colored_image_pred = colormap(normalized_data_pred / 255.0)  # Normalize to 0-1 for colormap

    # Convert the RGBA image to RGB by removing the alpha channel
    colored_image_rgb_gt = (colored_image_gt[:, :, :3] * 255).astype(np.uint8)
    colored_image_rgb_pred = (colored_image_pred[:, :, :3] * 255).astype(np.uint8)

    # Save the result as a PNG file
    output_file_gt = "biomass_raster_map_gt.png"
    output_file_pred = "biomass_raster_map_pred.png"
    plt.imsave(output_file_gt, colored_image_rgb_gt)
    plt.imsave(output_file_pred, colored_image_rgb_pred)

    print(f"Biomass raster map ground truth saved as {output_file_gt}")
    print(f"Biomass raster map saved as {output_file_pred}")

def evaluate_rmse_mae(LIST_OF_GT):
    
    norm_val = 256 * 256
    m = len(LIST_OF_GT)
    rmse = 0
    mae = 0
    mean_per_pix = 0

    for f in LIST_OF_GT:
        with Image.open(f"data/test_agbm/{f}") as img:
            raster_gt = np.array(img)
        with Image.open(f"pred_50ep/{f}") as img:
            raster_pred = np.array(img)

        rmse += np.sqrt(np.sum(np.square(np.subtract(raster_gt, raster_pred)))/norm_val)
        mean_per_pix += np.mean(raster_gt)
        mae += np.sum(np.abs(np.subtract(raster_gt, raster_pred)))/norm_val

    rmse /= m
    mean_per_pix /= m
    mae /= m

    return rmse, mae, mean_per_pix

rmse, mae, mpp = evaluate_rmse_mae(LIST_OF_FILES)
print(f"Root Mean Square Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
print(f"RMSE Percentage: {100*(rmse/mpp)}")