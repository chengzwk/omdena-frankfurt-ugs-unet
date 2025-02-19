import os
import json
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import seaborn as sns
from natsort import natsorted

# Define paths
data_dir = os.path.expanduser("~/Documents/Omdena/FrankfurtGermanyChapter_UrbanGreenSpaceMappping/MULC")

# Classes for segmentation
# 12 : Impervious surfaces and buildings
# 3: Low vegetation
# 4: Trees
# 5: Water
# 6: Clutter/Background
class_names = ["impervious", "low_veg", "tree", "water", "background"]

def compute_class_ratios(mask):
    """Function to compute class ratios for each class"""
    ratios = {class_name: np.mean(mask[i]) for i, class_name in enumerate(class_names)}
    return ratios

def classify_urban_environment(ratios):
    """Urban environment classification thresholds"""
    if ratios['impervious'] > 0.45:
        return 'Urban'
    elif ratios['water'] > 0.3:
        return 'River and Riverside'
    elif 0.3 < ratios['impervious'] <= 0.45 and (ratios['tree'] > 0.1 or ratios['low_veg'] > 0.1):
        return 'Suburban'
    elif ratios['tree'] > 0.6:
        return 'Forest'
    elif ratios['low_veg'] > 0.3 or ratios['tree'] > 0.3 or ratios['impervious'] < 0.2:
        return 'Outskirts/Agriculture'
    else:
        return 'Outskirts/Agriculture'

def normalize_by_layer(image_array):
    """
    Function to normalize image data to the same max(1) and min(0)
    Since different layers(bands) have different scales, normalization will be done layer by layer
    """
    # Normalize by band
    image_array = image_array.astype(np.float64)  # Convert dtype of image file from int to float64

    for i in range(image_array.shape[2]):
        layer_min = np.min(image_array[:, :, i])
        layer_max = np.max(image_array[:, :, i])

        try:
            image_array[:, :, i] = (image_array[:, :, i] - layer_min) / (layer_max - layer_min)
        except ZeroDivisionError:
            print(f"Band {i} has zero variation (min = max = {layer_min}). Skipping normalization.")
            image_array[:, :, i] = 0  # Set the band to default value 0

    return image_array

def plot_samples(df, num_samples=5):
    """ Plot random images with their masks and computed values """
    samples = df.sample(num_samples, random_state=42)

    for _, row in samples.iterrows():
        with rasterio.open(row["image_path"]) as src_img, rasterio.open(row["mask_path"]) as src_mask:
            image_array = src_img.read().transpose(1, 2, 0)
            mask_array = src_mask.read().transpose(1, 2, 0)

        image_array = image_array[:, :, (1, 2, 3)]          # get the bands you want; (1, 2, 3) is RGB bands
        image_array = normalize_by_layer(image_array)       # Normalize the image by band
        mask_array = np.argmax(mask_array, axis=2, keepdims=True) + 2  # Convert fractional mask to binary mask

        # Define class colors
        class_colors = {
            2: "#808080",  # Gray
            3: "#ADFF2F",  # Light Green
            4: "#006400",  # Dark Green
            5: "#1E90FF",  # Blue
            6: "#8B4513",  # Brown
        }

        cmap = plt.cm.colors.ListedColormap([class_colors[c] for c in sorted(class_colors)])
        norm = plt.cm.colors.BoundaryNorm(boundaries=[1.5, 2.5, 3.5, 4.5, 5.5, 6.5], ncolors=5)  # Use a discrete colormap

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(image_array)
        axes[0].set_title(f"Image: {row['image_file']} ({row['urban_type']})")

        axes[1].imshow(mask_array, cmap=cmap, norm=norm)
        axes[1].set_title(f"Mask: {row['urban_type']}")

        plt.figtext(0.5, 0.01, f"Ratios: {row[class_names].to_dict()}", ha="center")
        plt.show()

def compute_dataset_summary(image_dir):
    image_dir_data = []
    mask_dir = image_dir + '_masks'
    image_files = [f for f in os.listdir(os.path.join(data_dir, image_dir)) if f.endswith('GeoTIFF.tif')]
    image_files = natsorted(image_files)

    for image_file in image_files:
        mask_file = image_file.replace('.tif', '_fractional_mask.tif')
        image_path = os.path.join(data_dir, image_dir, image_file)
        mask_path = os.path.join(data_dir, mask_dir, mask_file)

        with rasterio.open(image_path) as img:
            image = img.read()
        with rasterio.open(mask_path) as msk:
            mask = msk.read()

        # Compute class ratios, spectral indices, and urban environment type
        ratios = compute_class_ratios(mask)
        urban_type = classify_urban_environment(ratios)

        image_dir_data.append({
            "image_file": image_file,
            "mask_file": mask_file,
            "image_path": image_path,
            "mask_path": mask_path,
            **ratios,
            "urban_type": urban_type
        })

    df = pd.DataFrame(image_dir_data)
    plot_samples(df)  # Inspect samples
    df.to_csv('dataset_overview_{}.csv'.format(image_dir), index=False)
    print("Dataset summary saved to dataset_overview_{}.csv. Total images: {}"
          .format(image_dir, len(df)))


# Load dataset and compute statistics for each dataset
image_dirs = sorted([d.replace('_masks', '') for d in os.listdir(data_dir) if d.endswith('_masks')])

for image_dir in image_dirs:
    if not os.path.exists(os.path.join(os.getcwd(), 'dataset_overview_{}.csv'.format(image_dir))):
        compute_dataset_summary(image_dir)

    # df = pd.read_csv('dataset_overview_{}.csv'.format(image_dir))
    # plot_samples(df, 15)

# Use Jupyter notebook for dataset inspection and statistics



