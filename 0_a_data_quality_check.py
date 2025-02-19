# Import packages
import os
import logging
import rasterio
import sys
import numpy as np
import random
import matplotlib.pyplot as plt
from natsort import natsorted


# Define paths
data_dir = os.path.expanduser("~/Documents/Omdena/FrankfurtGermanyChapter_UrbanGreenSpaceMappping/MULC")
log_dir = os.getcwd()

# Set up logging configuration
log_file = os.path.join(log_dir, "data_quality_check.log")

logging.basicConfig(filename=log_file,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode="w",
                    force=True)

class SuppressHTTPRequests(logging.Filter):
    def filter(self, record):
        return "HTTP Request: GET" not in record.getMessage()

# Get the root logger
logger = logging.getLogger()

# Apply the filter to suppress unwanted log messages
logger.addFilter(SuppressHTTPRequests())

# Apply the filter to all other loggers (to cover external libraries)
for name in logging.root.manager.loggerDict:
    logging.getLogger(name).addFilter(SuppressHTTPRequests())

# Debugging: Check where the log file is being saved
print(f"Logging to: {log_file}")


def print_progress_bar(iteration, total, length=40):
    """Print a progress bar."""
    percent = (iteration / total) * 100
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent:.2f}%')
    sys.stdout.flush()

def data_consistency_check(image_dir):
    mask_dir = image_dir + '_masks'
    image_files = [f for f in os.listdir(os.path.join(data_dir, image_dir)) if f.endswith('GeoTIFF.tif')]
    image_files = natsorted(image_files)
    total_files = len(image_files)

    # Print and log progress
    print(f"Checking {image_dir} and {mask_dir}")
    logging.info(f"Checking {image_dir} and {mask_dir}")

    # Loop through all image files in image_dir
    for i, image_file in enumerate(image_files):

        mask_file = image_file.replace('.tif', '_fractional_mask.tif')
        image_path = os.path.join(data_dir, image_dir, image_file)
        mask_path = os.path.join(data_dir, mask_dir, mask_file)

        # Verify the corresponding mask exists
        if not os.path.exists(mask_path):
            logging.warning(f'Mask file does not exist for {image_file}. Expected at {mask_path}.')
            continue  # Skip to the next image if the mask is missing

        try:
            # Open image and mask
            with rasterio.open(image_path) as img:
                image = img.read()
                image_meta = img.meta

            with rasterio.open(mask_path) as msk:
                mask = msk.read()
                mask_meta = msk.meta

            # Verify the size of image and mask is 128x128 pixels
            if image.shape[1:] != (128, 128):
                logging.error(f'Image {image_file} is not 128x128 pixels. Found shape: {image.shape[1:]}')

            if mask.shape[1:] != (128, 128):
                logging.error(f'Mask {mask_file} is not 128x128 pixels. Found shape: {mask.shape[1:]}')

            # Verify that the image and mask have the same CRS and transform
            if img.crs != msk.crs:
                logging.error(f'CRS mismatch for {image_file} and {mask_file}: {img.crs} vs {msk.crs}')

            if img.transform != msk.transform:
                logging.error(f'Transform mismatch for {image_file} and {mask_file}: {img.transform} vs {msk.transform}')

            # Verify that the fractions in the mask file sum up to 1
            epsilon = 0.01
            npixel = np.count_nonzero(np.abs(1.0 - np.sum(mask, axis=0)) >= epsilon)
            if npixel != 0:
                logging.error(f'Fractions in mask for {image_file} and {mask_file} doesn\'t sum up to 1. Found {npixel} pixels.')

        except Exception as e:
            logging.error(f'Error processing {image_file}: {e}')
            continue  # Skip to next image if an error occurs

        # Print progress
        print_progress_bar(i + 1, total_files)  # Update progress bar

    # Print and log progress
    print(f"\nCompleted checks for {image_dir} and {mask_dir}")
    logging.info(f"Completed checks for {image_dir} and {mask_dir}")

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

def plot_samples(image_dir, num_samples=5):
    """ Plot random images with their masks and computed values """
    mask_dir = image_dir + '_masks'
    image_files = [f for f in os.listdir(os.path.join(data_dir, image_dir)) if f.endswith('GeoTIFF.tif')]
    image_files = natsorted(image_files)
    selected_files = random.sample(image_files, min(num_samples, len(image_files)))

    for image_file in selected_files:
        mask_file = image_file.replace('.tif', '_fractional_mask.tif')
        image_path = os.path.join(data_dir, image_dir, image_file)
        mask_path = os.path.join(data_dir, mask_dir, mask_file)

        with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:
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
        axes[0].set_title(f"Image: {image_file}")

        axes[1].imshow(mask_array, cmap=cmap, norm=norm)
        axes[1].set_title("Mask")

        plt.show()


# Load dataset and compute statistics for each dataset
image_dirs = sorted([d.replace('_masks', '') for d in os.listdir(data_dir) if d.endswith('_masks')])

# Loop through all images directories and run the check function
for image_dir in image_dirs:
    if not os.path.exists(log_file):
        data_consistency_check(image_dir)
    plot_samples(image_dir)





