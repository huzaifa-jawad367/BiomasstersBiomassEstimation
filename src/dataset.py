import random

import cv2
import numpy as np
import torch
from skimage import io
import pandas as pd
from pathlib import Path


import xarray as xr
from xrspatial import multispectral

L = 0.5

s1_min = np.array([-25 , -62 , -25, -60], dtype="float32")
s1_max = np.array([ 29 ,  28,  30,  22 ], dtype="float32")
s1_mm = s1_max - s1_min

s2_max = np.array(
    [19616., 18400., 17536., 17097., 16928., 16768., 16593., 16492., 15401., 15226., 1., 10316., 8406., 1., 255.],
    dtype="float32",
)

# s2_max = np.array(
#     [19616., 18400., 17536., 17097., 16928., 16768., 16593., 16492., 15401., 15226., 255.],
#     dtype="float32",
# )

IMG_SIZE = (256, 256)

epsilon = 1e-6  # Small constant to avoid division by zero

def calculate_veg_indices_uint8(img_s2):
    """
    Calculate vegetation indices and convert them to uint8.
    Args:
        img_s2 (np.ndarray): Sentinel-2 image array (H, W, Bands)
    
    Returns:
        dict: A dictionary of vegetation indices scaled to uint8
    
    # Calculate vegetation indices
    ndvi = (img_s2[:, :, 6] - img_s2[:, :, 2]) / (img_s2[:, :, 6] + img_s2[:, :, 2] + epsilon)
    evi = (2.5 * (img_s2[:, :, 6] - img_s2[:, :, 2])) / (
        img_s2[:, :, 6] + 6 * img_s2[:, :, 2] - 7.5 * img_s2[:, :, 0] + 1 + epsilon
    )
    savi = ((img_s2[:, :, 6] - img_s2[:, :, 2]) * (1 + 0.5)) / (img_s2[:, :, 6] + img_s2[:, :, 2] + 0.5 + epsilon)
    msavi = 0.5 * (
        2 * img_s2[:, :, 6] + 1 - np.sqrt(
            np.square(2 * img_s2[:, :, 6] + 1) - 8 * (img_s2[:, :, 6] - img_s2[:, :, 2]) + epsilon
        )
    )
    ndmi = (img_s2[:, :, 6] - img_s2[:, :, 7]) / (img_s2[:, :, 6] + img_s2[:, :, 7] + epsilon)
    nbr = (img_s2[:, :, 6] - img_s2[:, :, 8]) / (img_s2[:, :, 6] + img_s2[:, :, 8] + epsilon)
    nbr2 = (img_s2[:, :, 7] - img_s2[:, :, 8]) / (img_s2[:, :, 7] + img_s2[:, :, 8] + epsilon)

    """
    img_s2_xr = xr.DataArray(img_s2)

    # Define epsilon to avoid division by zero
    epsilon = 1e-6
    
    # NDVI - Normalized Difference Vegetation Index
    ndvi = np.array(multispectral.ndvi(img_s2_xr[:, :, 6], img_s2_xr[:, :, 2]))
    
    # EVI - Enhanced Vegetation Index
    evi = np.array(multispectral.evi(img_s2_xr[:, :, 6], img_s2_xr[:, :, 2], img_s2_xr[:, :, 0]))
    
    # Removing SAVI due to irrelevance
    # # SAVI - Soil-Adjusted Vegetation Index
    # savi = np.array(multispectral.savi(img_s2_xr[:, :, 6], img_s2_xr[:, :, 2]))
    
    # # MSAVI - Modified Soil-Adjusted Vegetation Index
    # msavi = multispectral.msavi(img_s2_xr[:, :, 6], img_s2_xr[:, :, 2])
    msavi = 0.5 * (
        2 * img_s2[:, :, 6] + 1 - np.sqrt(
            np.square(2 * img_s2[:, :, 6] + 1) - 8 * (img_s2[:, :, 6] - img_s2[:, :, 2])
        )
    )
    
    # NDMI - Normalized Difference Moisture Index
    ndmi = np.array(multispectral.ndmi(img_s2_xr[:, :, 6], img_s2_xr[:, :, 7]))
    
    # Remove NBR because of irrelevance
    # # NBR - Normalized Burn Ratio
    # nbr = np.array(multispectral.nbr(img_s2_xr[:, :, 6], img_s2_xr[:, :, 8]))
    
    # # NBR2 - Another variation of Normalized Burn Ratio
    # nbr2 = np.array(multispectral.nbr2(img_s2_xr[:, :, 7], img_s2_xr[:, :, 8]))

    # Normalize indices to [0, 255] and convert to uint8
    def normalize_and_convert(index):
        index_normalized = (index + 1) / 2  # Scale [-1, 1] to [0, 1]
        # index_uint8 = (index_normalized * 255).clip(0, 255).astype("uint8")  # Scale to [0, 255] and convert to uint8
        return index_normalized

    # Create a dictionary of uint8 vegetation indices
    indices_uint8 = {
        "ndvi": normalize_and_convert(ndvi),
        "evi": normalize_and_convert(evi),
        # "savi": normalize_and_convert(savi),
        "msavi": normalize_and_convert(msavi),
        "ndmi": normalize_and_convert(ndmi),
        # "nbr": normalize_and_convert(nbr),
        # "nbr2": normalize_and_convert(nbr2),
    }

    return indices_uint8


def read_imgs(chip_id, data_dir, veg_indices=False):
    # Ensure data_dir is a Path object
    data_dir = Path(data_dir) if isinstance(data_dir, str) else data_dir
    imgs, mask = [], []
    for month in range(12):
        img_s1 = io.imread(data_dir / f"{chip_id}_S1_{month:0>2}.tif")
        m = img_s1 == -9999
        img_s1 = img_s1.astype("float32")
        img_s1 = (img_s1 - s1_min) / s1_mm
        img_s1 = np.where(m, 0, img_s1)
        filepath = data_dir / f"{chip_id}_S2_{month:0>2}.tif"
        if filepath.is_file():
            img_s2 = io.imread(filepath)
            img_s2 = img_s2.astype("float32")

            main_channels = img_s2[:, :, :-1]
            transparency_channel = img_s2[:, :, -1:]

            if veg_indices:
                veg_indices_uint8 = calculate_veg_indices_uint8(img_s2)
                img_s2 = main_channels
                for index_name, index_array in veg_indices_uint8.items():
                    if np.isnan(index_array).any():
                        index_array = np.nan_to_num(index_array, nan=0.0)
                    index_array = np.expand_dims(index_array, axis=2)
                    img_s2 = np.concatenate([img_s2, index_array], axis=2)
                    # print(f"{index_name} max: {np.max(index_array)}, {np.count_nonzero(np.isnan(index_array))}")

                img_s2 = np.concatenate([img_s2, transparency_channel], axis=2)
                # print(f"Before Normalisation: {np.max(img_s2)}")

            img_s2 = img_s2 / s2_max

            # print(f"After Normalisation: {np.max(img_s2)}")
            
        else:
            img_s2 = np.zeros(IMG_SIZE + (15,), dtype="float32")
            # img_s2 = np.zeros(IMG_SIZE + (11,), dtype="float32")

        img = np.concatenate([img_s1, img_s2], axis=2)
        img = np.transpose(img, (2, 0, 1))
        imgs.append(img)
        mask.append(False)

    mask = np.array(mask)

    imgs = np.stack(imgs, axis=0)  # [t, c, h, w]

    return imgs, mask


def rotate_image(image, angle, rot_pnt, scale=1):
    rot_mat = cv2.getRotationMatrix2D(rot_pnt, angle, scale)
    result = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101) #INTER_NEAREST

    return result


def train_aug(imgs, mask, target):
    # imgs: [t, c, h, w]
    # mask: [t, ]
    # target: [h, w]
    if random.random() > 0.5:  # horizontal flip
        imgs = imgs[..., ::-1]
        target = target[..., ::-1]

    k = random.randrange(4)
    if k > 0:  # rotate90
        imgs = np.rot90(imgs, k=k, axes=(-2, -1))
        target = np.rot90(target, k=k, axes=(-2, -1))

    if random.random() > 0.3:  # scale-rotate
        _d = int(imgs.shape[2] * 0.1)  # 0.4)
        rot_pnt = (imgs.shape[2] // 2 + random.randint(-_d, _d), imgs.shape[3] // 2 + random.randint(-_d, _d))
        #scale = 1
        #if random.random() > 0.2:
            #scale = random.normalvariate(1.0, 0.1)

        #angle = 0
        #if random.random() > 0.2:
        angle = random.randint(0, 90) - 45

        if (angle != 0): # or (scale != 1):
            t = len(imgs)  # t, c, h, w
            imgs = np.concatenate(imgs, axis=0)  # t*c, h, w
            imgs = np.transpose(imgs, (1, 2, 0))  # h, w, t*c
            imgs = rotate_image(imgs, angle, rot_pnt)
            imgs = np.transpose(imgs, (2, 0, 1))  # t*c, h, w
            imgs = np.reshape(imgs, (t, -1, imgs.shape[1], imgs.shape[2]))  # t, c, h, w
            target = rotate_image(target, angle, rot_pnt)

    if random.random() > 0.5:  # "word" dropout
        while True:
            mask2 = np.random.rand(*mask.shape) < 0.3
            mask3 = np.logical_or(mask, mask2)
            if not mask3.all():
                break

        mask = mask3
        imgs[mask2] = 0

    return imgs.copy(), mask, target.copy()


class DS(torch.utils.data.Dataset):
    def __init__(self, df, dir_features, dir_labels=None, augs=False, veg_indices=False):
        self.df = df
        self.dir_features = dir_features
        self.dir_labels = dir_labels
        self.augs = augs
        self.veg_indices = veg_indices

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]

        imgs, mask = read_imgs(item.chip_id, self.dir_features, self.veg_indices)
        if self.dir_labels is not None:
            target = io.imread(self.dir_labels / f'{item.chip_id}_agbm.tif')
        else:
            target = item.chip_id

        if self.augs:
            imgs, mask, target = train_aug(imgs, mask, target)

        return imgs, mask, target


def predict_tta(models, images, masks, ntta=1):
    result = images.new_zeros((images.shape[0], 1, images.shape[-2], images.shape[-1]))
    n = 0
    for model in models:
        logits = model(images, masks)
        result += logits
        n += 1

        if ntta == 2:
            # hflip
            logits = model(torch.flip(images, dims=[-1]), masks)
            result += torch.flip(logits, dims=[-1]) 
            n += 1

        if ntta == 3:
            # vflip
            logits = model(torch.flip(images, dims=[-2]), masks)
            result += torch.flip(logits, dims=[-2])
            n += 1

        if ntta == 4:
            # hvflip
            logits = model(torch.flip(images, dims=[-2, -1]), masks)
            result += torch.flip(logits, dims=[-2, -1])
            n += 1

    result /= n * len(models)

    return result

if __name__ == "__main__":
    df = pd.read_csv("data/features_metadata.csv")
    test_df = df[df.split == "test"].copy()
    test_df = test_df.groupby("chip_id").agg(list).reset_index()
    print(test_df)

    test_images_dir = 'data/test_features'
    test_dataset = DS(
        df=test_df,
        dir_features=test_images_dir,
        veg_indices=True
    )

    for idx in range(len(test_dataset)):
        imgs, masks, target = test_dataset[idx]
        print(f"Data at index {idx}:", imgs.shape)

        
