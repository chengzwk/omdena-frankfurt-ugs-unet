import os
import json
import numpy as np
import random
import rasterio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Define paths
# Use absolute path for data directory; Change to your data directory path
data_dir = "~/Documents/Omdena/FrankfurtGermanyChapter_UrbanGreenSpaceMappping/MULC"
mask_pattern = "*_GeoTIFF_fractional_mask.tif"

# Classes for segmentation
# 12 : Impervious surfaces and buildings
# 3: Low vegetation
# 4: Trees
# 5: Water
# 6: Clutter/Background
class_names = {"impervious", "low_veg", "tree", "water", "background"}

# Urban environment classification thresholds
def classify_urban_environment(ratios):
    if ratios['impervious'] > 0.7:
        return 'Dense Urban'
    elif 0.3 < ratios['impervious'] <= 0.7 and ratios['tree'] > 0.1:
        return 'Suburban'
    elif ratios['water'] > 0.3:
        return 'River and Riverside'
    elif ratios['tree'] > 0.6:
        return 'Forest'
    elif ratios['low_veg'] > 0.3 or ratios['impervious'] < 0.2:
        return 'Outskirts/Agriculture'
    else:
        return 'Unknown'

# Function to compute statistics
def compute_class_ratios(mask_path):
    with rasterio.open(mask_path) as src:
        mask = src.read()  # Shape: (5, H, W)
    ratios = {"impervious": np.mean(mask[0]),
              "low_veg": np.mean(mask[1]),
              "tree": np.mean(mask[2]),
              "water": np.mean(mask[3]),
              "background": np.mean(mask[4])}
    return ratios

# Load dataset metadata
#### existing meta data from dagshub engine?
image_chips = []
for root, _, files in os.walk(data_dir):
    for file in files:
        if file.endswith("_GeoTIFF_fractional_mask.tif"):
            mask_path = os.path.join(root, file)
            ratios = compute_class_ratios(mask_path)
            urban_type = next((k for k, v in urban_classes.items() if v(ratios)), "Unknown")
            image_chips.append({"file": file, "path": mask_path, "ratios": ratios, "urban_type": urban_type})

# Compute dataset statistics
#### add visualization (histogram, boxplot)
ratios_list = [chip["ratios"] for chip in image_chips]
stats = {class_name: {"mean": np.mean([r[class_name] for r in ratios_list]),
                      "std": np.std([r[class_name] for r in ratios_list]),
                      "median": np.median([r[class_name] for r in ratios_list])}
         for class_name in class_names}

# Split dataset maintaining distribution
#### need explanation, look up train_test_split
urban_types = [chip["urban_type"] for chip in image_chips]
train, test = train_test_split(image_chips, test_size=0.2, stratify=urban_types, random_state=42)
train, val = train_test_split(train, test_size=0.1, stratify=[chip["urban_type"] for chip in train], random_state=42)

# Save metadata
#### save in dagshub format
metadata = {"train": train, "val": val, "test": test, "stats": stats}
with open("dataset_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# Manual inspection: Plot some samples
# add satellite image for contrast, plot binary mask also
def plot_samples(samples, title):
    fig, axes = plt.subplots(1, len(samples), figsize=(15, 5))
    for ax, sample in zip(axes, samples):
        with rasterio.open(sample["path"]) as src:
            mask = src.read()  # (5, H, W)
        ax.imshow(mask[0], cmap="gray")
        ax.set_title(sample["urban_type"])
    plt.suptitle(title)
    plt.show()

for urban_type in urban_classes.keys():
    samples = [chip for chip in image_chips if chip["urban_type"] == urban_type][:5]
    if samples:
        plot_samples(samples, f"Sample Masks - {urban_type}")

print("Dataset processing complete. Metadata saved as 'dataset_metadata.json'.")



import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

def plot_random_samples(image_dirs, dataset_dir, samples_per_dir=2):
    """
    Randomly selects images from multiple directories, reads corresponding masks,
    converts fractional masks to binary, and plots them.

    Parameters:
    - image_dirs: List of image directories
    - dataset_dir: Root dataset directory
    - samples_per_dir: Number of samples to select per image directory
    """
    # Define color mapping for mask visualization
    class_colors = {
        2: "#808080",  # Gray (impervious)
        3: "#ADFF2F",  # Light Green (low vegetation)
        4: "#006400",  # Dark Green (trees)
        5: "#1E90FF",  # Blue (water)
        6: "#8B4513",  # Brown (clutter)
    }

    for i in image_dirs:
        image_dir = os.path.join(dataset_dir, i)
        mask_dir = os.path.join(dataset_dir, i + '_masks')

        # Get list of image files
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]
        selected_files = random.sample(image_files, min(samples_per_dir, len(image_files)))

        for img_file in selected_files:
            img_full_path = os.path.join(image_dir, img_file)
            mask_file = img_file.replace(".tif", "_fractional_mask.tif")
            mask_full_path = os.path.join(mask_dir, mask_file)

            # Read and preprocess image
            with open(img_full_path, "rb") as f:
                with rasterio.open(f) as img:
                    image = img.read()
            image = np.transpose(image, [1, 2, 0])  # Move bands to last axis
            image[np.isnan(image)] = 0  # Replace NaN with 0
            image = image[:, :, (1, 2, 3)]  # Select RGB bands
            image = normalize_by_layer(image)

            # Read and preprocess mask
            with open(mask_full_path, "rb") as m:
                with rasterio.open(m) as msk:
                    mask = msk.read()
            mask = np.transpose(mask, [1, 2, 0])  # Move bands to last axis
            mask[np.isnan(mask)] = 0  # Replace NaN with 0
            mask = np.argmax(mask, axis=2, keepdims=True) + 2  # Convert to binary classes

            # Plot Image and Mask
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(image)
            axes[0].set_title(f"Image: {img_file}")

            cmap = plt.cm.colors.ListedColormap([class_colors[c] for c in sorted(class_colors)])
            axes[1].imshow(mask[:, :, 0], cmap=cmap)
            axes[1].set_title(f"Mask: {mask_file}")

            # Create a legend
            legend_patches = [mpatches.Patch(color=class_colors[key], label=f"Class {key}") for key in class_colors]
            fig.legend(handles=legend_patches, loc="center left", bbox_to_anchor=(1.01, 0.5), title="Legend")

            plt.subplots_adjust(right=0.95)
            plt.show()


# Randomly select 2 images from each image directory,
# plot the image and corresponding mask
plot_random_samples(image_dirs, dataset_dir=dataset_dir, samples_per_dir=2)


def iou(y_true, y_pred):
    """Calculates Intersection over Union (IoU) / Jaccard Index."""
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3]) - intersection
    return K.mean((intersection + K.epsilon()) / (union + K.epsilon()), axis=0)

def f1_score(y_true, y_pred):
    """Calculates F1 Score (harmonic mean of precision and recall)."""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=[1, 2, 3])
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=[1, 2, 3])
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=[1, 2, 3])
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return K.mean(2 * (precision * recall) / (precision + recall + K.epsilon()))

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=BinaryCrossentropy(from_logits=True),  # Pixel-wise binary cross-entropy loss
              metrics=['accuracy', iou, f1_score])
