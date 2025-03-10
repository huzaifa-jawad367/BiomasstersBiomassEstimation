# Re-import necessary libraries after execution reset
import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define directories
test_agbm_dir = "data/test_agbm"
pred_agbm_dir = "predictions/pred2"

min_agg_rmse = 150

# Verify that directories exist before proceeding
if not os.path.exists(test_agbm_dir) or not os.path.exists(pred_agbm_dir):
    raise FileNotFoundError("One or both of the specified directories do not exist. Please verify the paths.")

# Get list of files
list_of_files = os.listdir(test_agbm_dir)

# Initialize RMSE sum
Sum_rmse = 0

# Ensure the output directory exists
os.makedirs("error_maps/pred2", exist_ok=True)

# Function to compute RMSE
def compute_rmse(test_array, pred_array):
    return np.sqrt(np.mean((test_array - pred_array) ** 2))

# Function to plot and save test, predicted, and RMSE maps
def plot_raster_maps(test_array, pred_array, rmse_map, filename, aggregated_rmse):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot test map
    im1 = axes[0].imshow(test_array, cmap='viridis')
    axes[0].set_title("Test Raster Map")
    plt.colorbar(im1, ax=axes[0])
    
    # Plot predicted map
    im2 = axes[1].imshow(pred_array, cmap='viridis')
    axes[1].set_title("Predicted Raster Map")
    plt.colorbar(im2, ax=axes[1])
    
    # Plot RMSE error map
    im3 = axes[2].imshow(rmse_map, cmap='Reds')
    axes[2].set_title("RMSE Error Map")
    plt.colorbar(im3, ax=axes[2])
    
    # Set the title
    fig.suptitle(f"File: {filename} | Aggregated RMSE: {aggregated_rmse:.4f}", fontsize=12)
    
    # Adjust layout
    plt.tight_layout()

    # Save the plot
    save_path = f"error_maps/pred2_max/min_error.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    
    # Close the plot to free memory
    plt.close(fig)

# Process each file with tqdm progress bar
for file in tqdm(list_of_files, desc="Processing files", unit="file"):
    pred_file_path = os.path.join(pred_agbm_dir, file)
    test_file_path = os.path.join(test_agbm_dir, file)
    
    # Open the raster files
    with rasterio.open(test_file_path) as test_dataset, rasterio.open(pred_file_path) as pred_dataset:
        test_array = test_dataset.read(1)  # Read first band
        pred_array = pred_dataset.read(1)  # Read first band

        # Compute RMSE map
        rmse_map = np.sqrt((test_array - pred_array) ** 2)
        aggregated_rmse = compute_rmse(test_array, pred_array)

        if aggregated_rmse < min_agg_rmse:
            # Plot and save results
            min_agg_rmse = aggregated_rmse
            plot_raster_maps(test_array, pred_array, rmse_map, file, aggregated_rmse)

        Sum_rmse += aggregated_rmse

        # # Plot and save results
        # plot_raster_maps(test_array, pred_array, rmse_map, file, aggregated_rmse)

# Print final aggregated RMSE over all images
average_rmse = Sum_rmse / len(list_of_files)
print(f"Average RMSE over {len(list_of_files)} images: {average_rmse:.4f}")
