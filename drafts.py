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
