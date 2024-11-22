import os
import numpy as np
import tifffile as tiff
import torch
import json

def checking_files():
    list_of_files = os.listdir('train_features')
    for sample_file in list_of_files:
        path_to_tiff_file = f'train_features/{sample_file}'
        print(path_to_tiff_file)
        try:
            # Check if the file exists
            if not os.path.exists(path_to_tiff_file):
                raise FileNotFoundError(f"File does not exist: {path_to_tiff_file}")

            # Attempt to read the TIFF file
            img = tiff.imread(path_to_tiff_file)
            print(img.shape)  # Print the dimensions of the image

        except FileNotFoundError as e:
            print(e)
        except tiff.TiffFileError as e:
            print(f"Error reading TIFF file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        break


# # Path to your .pth file
# pth_file = "models/tf_efficientnetv2_l_in21k_f0_b4x1_e20_nrmse_devscse_attnlin_augs_decplus7/modelo_best.pth"
# json_file = "model_architecture.json"

# # Load the .pth file
# checkpoint = torch.load(pth_file)

# # Extract model architecture
# # Check if it contains a model object
# if 'model' in checkpoint:
#     model = checkpoint['model']
#     architecture = str(model)  # Convert model to string for readability
# elif 'state_dict' in checkpoint:
#     state_dict = checkpoint['state_dict']
#     # Extract architecture information from state_dict keys
#     architecture = {
#         "layers": list(state_dict.keys())
#     }
# else:
#     raise ValueError("The .pth file does not contain a model or state_dict.")

# # Save to JSON
# with open(json_file, 'w') as f:
#     json.dump(architecture, f, indent=4)

# print(f"Model architecture saved to {json_file}")

with open('model_details.txt', 'r') as fhand:
    my_dict = fhand.readlines()
    print(my_dict[1])

