import pandas as pd
from dataset import *
import matplotlib.pyplot as plt

def save_img(img_array):
    norm_img = (255 * (img_array - img_array.min()) / (img_array.max() - img_array.min())).astype(np.uint8)
    plt.imsave('SAR_image.png', norm_img)


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
        img, mask, target = test_dataset[idx]
        print(f"Data at index {idx}:", img.shape)
        sar_img = img[0][11:15].reshape((256,256,4))
        break