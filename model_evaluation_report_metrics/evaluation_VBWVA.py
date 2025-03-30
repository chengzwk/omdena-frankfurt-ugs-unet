import os
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import BinaryFocalCrossentropy
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score

from data_preparation import read_file, split_dataset, formatted_print_shapes
from model_evaluation_prediction import predict, evaluate_model, plot_roc_curve, evaluate_model_sklearn, visualize_predictions, visualize_prediction_comparison
from visualize_models_evaluation_results import visualize_eval_results


# --- Reload VBWVA test set ---
data_dir = os.path.expanduser("~/Documents/Omdena/FrankfurtGermanyChapter_UrbanGreenSpaceMappping/MULC")
image_dir = 'VBWVA_8R'
bands = ['Blue', 'Green', 'Red', 'NIR', 'NDVI', 'NDWI']
image_dataset, mask_dataset = read_file(
    data_dir,
    image_dir,
    multiclass=False,
    selected_bands=bands
)

# Train-validation-test split
X_train, X_val, X_test_allbands, y_train, y_val, y_test = split_dataset(image_dataset, mask_dataset)
del image_dataset, mask_dataset
formatted_print_shapes(X_train, X_val, X_test_allbands, y_train, y_val, y_test)

# --- Evaluate U-Net models ---
# Define base directory
base_dir = os.path.expanduser("~/Documents/Omdena/band_combinations")
# Define model directories
model_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

# Dictionary to store evaluation results
eval_data = []
eval_results = []

# Loop through each model
for model_id in model_dirs:
    # Create test set with correct band combination
    model_bands = model_id.split('_')[1:]
    X_test = np.stack([X_test_allbands[:, :, :, i] for i, band in enumerate(bands) if band in model_bands], axis=-1)

    # Load model
    model_dir = os.path.join(base_dir, model_id)
    model_path = os.path.join(model_dir, "best_model_unet.keras")
    model = load_model(model_path, compile=True)
    print(f"Evaluating model: {model_id}")

    # Evaluate model on test set
    loss, acc, precision, recall, f1, mean_iou, iou_class1, auc, y_pred_thresholded = evaluate_model_sklearn(model, X_test,
                                                                                                     y_test)
    # Store evaluation data
    eval_data.append({
        "Combination of bands": '-'.join(model_id.split('_')[1:]),
        "X_test": X_test,
        "y_test": y_test,
        "y_pred_thresholded": y_pred_thresholded
    })

    # Store results
    eval_results.append({
        "Combination of bands": '-'.join(model_id.split('_')[1:]),
        "Loss": loss,
        "OA": acc,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "Mean IoU": mean_iou,
        "IoU (Veg)": iou_class1,
        "AUC": auc
    })

# Convert results to a DataFrame
df_eval = pd.DataFrame(eval_results)
# df_eval = df_eval.sort_values(by="OA", ascending=False)  # Sort by accuracy

# Visualize evaluation results
metrics = ["OA", "F1-score", "Mean IoU"]  # Select metrics for visualization
title = "Comparison of Model Performance on VBWVA test set"
visualize_eval_results(df_eval, metrics, title)

# Display results
print(df_eval)

# --- Visualize Model Prediction ---
# Select best model on OA to visualize
best_model = df_eval.loc[df_eval['OA'].idxmax(), 'Combination of bands']
print(f"Visualize prediction result for band combination {best_model}")

# Find the corresponding entry in eval_data
best_data = None
for data in eval_data:
    if data['Combination of bands'] == best_model:
        best_data = data
        break

X_test = best_data['X_test']
y_test = best_data['y_test']
y_pred_thresholded = best_data['y_pred_thresholded']

visualize_predictions(X_test, y_test, y_pred_thresholded,
                      title="Predictions on VBWVA test images",
                      num_samples=5)
visualize_prediction_comparison(X_test, y_test, y_pred_thresholded,
                                title="Comparison of Prediction and Ground Truth on VBWVA test images",
                                num_samples=5)
