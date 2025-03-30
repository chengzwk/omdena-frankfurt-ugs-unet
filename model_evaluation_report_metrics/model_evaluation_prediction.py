import numpy as np
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from fontTools.cffLib import encodeNumber
from tensorflow.keras.metrics import IoU, MeanIoU
from sklearn.metrics import roc_curve, auc
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, jaccard_score


def calculate_f1_score(precision, recall):
    """Calculates the F1 score and avoids division by zero."""
    denominator = precision + recall
    if denominator == 0:
        f1_score = 0.0
    else:
        f1_score = 2 * precision * recall / (precision + recall)
    return f1_score

def predict(model, X_test, threshold=0.5):
    """Generate predictions and apply thresholding."""
    y_pred = model.predict(X_test)
    # y_pred = model(X_test, training=True)
    y_pred = tf.sigmoid(y_pred).numpy()
    y_pred_thresholded = (y_pred >= threshold).astype(int)

    return y_pred, y_pred_thresholded

def evaluate_model(model, X_test, y_test, ifsilent=False):
    """Evaluate model on test set and return loss, accuracy, precision, and recall."""
    loss, acc, precision, recall, mean_iou, iou_class1 = model.evaluate(X_test, y_test, verbose=0)
    f1_score = calculate_f1_score(precision, recall)

    if not ifsilent:
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Mean IoU: {mean_iou:.4f}")
        print(f"IoU for class 1: {iou_class1:.4f}")

    return loss, acc, precision, recall, f1_score, mean_iou, iou_class1

def compute_iou(y_pred_thresholded, y_test):
    """Compute IoU for class 1 and Mean IoU."""
    iou_metric = IoU(num_classes=2, target_class_ids=[1])
    mean_iou_metric = MeanIoU(num_classes=2)

    iou_metric.update_state(y_pred_thresholded, y_test)
    mean_iou_metric.update_state(y_pred_thresholded, y_test)

    iou_value = iou_metric.result().numpy()
    mean_iou_value = mean_iou_metric.result().numpy()

    print(f"IoU for class 1: {iou_value:.4f}")
    print(f"Mean IoU: {mean_iou_value:.4f}")

    return iou_value, mean_iou_value

def plot_roc_curve(y_pred_thresholded, y_test, result_dir=None, ifplot=True):
    """Plot ROC curve and compute AUC."""
    y_test_raveled = y_test.ravel().astype(int)
    y_pred_thresholded_raveled = y_pred_thresholded.ravel().astype(int)

    fpr, tpr, thresholds = roc_curve(y_test_raveled, y_pred_thresholded_raveled)
    auc_value = auc(fpr, tpr)

    if ifplot:
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc_value:.4f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid()
        plt.savefig(os.path.join(result_dir, "ROC_curve.png"))

        print(f"AUC: {auc_value:.4f}")
    return auc_value

def visualize_predictions(X, y_true, y_pred, result_dir=None, title="Predictions vs. Ground Truth", num_samples=None):
    """Helper function to visualize predictions against ground truth."""
    if num_samples is None:
        num_samples = X.shape[0]
        sample_indices = range(len(X))
    else:
        sample_indices = random.sample(range(len(X)), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(10, 5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure consistent indexing

    for r, i in enumerate(sample_indices):
        axes[r, 0].imshow(X[i][:, :, :3])
        axes[r, 0].set_title("Input Image")
        axes[r, 0].axis("off")

        axes[r, 1].imshow(y_true[i].squeeze(), cmap="gray")
        axes[r, 1].set_title("Ground Truth Mask")
        axes[r, 1].axis("off")

        axes[r, 2].imshow(y_pred[i].squeeze(), cmap="gray")
        axes[r, 2].set_title("Predicted Mask")
        axes[r, 2].axis("off")

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    # plt.show()
    result_dir = os.getcwd()
    plt.savefig(os.path.join(result_dir, f"{title}.png"))
    plt.close()

def visualize_prediction_comparison(X, y_true, y_pred, result_dir=None, title="Prediction Comparison",
                                    num_samples=None):
    """
    Visualizes the comparison between the predicted mask and the ground truth.

    Green: Correctly predicted vegetation (intersection of prediction and ground truth).
    Red: False positives (predicted vegetation that is not in the ground truth).
    Blue: False negatives (missed vegetation in ground truth but not predicted).

    Parameters:
        X (numpy array): Input images.
        y_true (numpy array): Ground truth masks (binary).
        y_pred (numpy array): Predicted masks (binary).
        result_dir (str): Directory to save the visualization.
        title (str): Title of the plot.
    """
    if num_samples is None:
        num_samples = X.shape[0]
        sample_indices = range(len(X))
    else:
        sample_indices = random.sample(range(len(X)), num_samples)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure consistent indexing

    # Define color mapping (single-channel image with unique color values)
    COLOR_MAPPING = {
        0: (0, 0, 0),  # Black (Background)
        # 0: (220 / 255, 220 / 255, 220 / 255),  # Light Grey (Background)
        1: (76 / 255, 175 / 255, 80 / 255),  # Green (True Positive)
        2: (244 / 255, 67 / 255, 54 / 255),  # Red (False Positive)
        3: (255 / 255, 152 / 255, 0 / 255)  # Orange (False Negative)
    }

    for r, i in enumerate(sample_indices):
        # Compute error visualization
        comparison_mask = np.zeros_like(y_true[i], dtype=np.uint8)

        intersection = (y_true[i] == 1) & (y_pred[i] == 1)  # True Positive (Green)
        false_positive = (y_true[i] == 0) & (y_pred[i] == 1)  # False Positive (Red)
        false_negative = (y_true[i] == 1) & (y_pred[i] == 0)  # False Negative (Orange)

        comparison_mask[intersection] = 1
        comparison_mask[false_positive] = 2
        comparison_mask[false_negative] = 3

        # Create a custom colormap
        colormap = np.zeros((4, 3))  # 4 unique values (0-3)
        for key, color in COLOR_MAPPING.items():
            colormap[key] = color

        # Plot original image
        axes[r, 0].imshow(X[i][:, :, :3])  # Show only RGB channels
        axes[r, 0].set_title("Input Image")
        axes[r, 0].axis("off")

        # Plot ground truth
        axes[r, 1].imshow(y_true[i].squeeze(), cmap="gray")
        axes[r, 1].set_title("Ground Truth Mask")
        axes[r, 1].axis("off")

        # Plot comparison mask with assigned colors
        axes[r, 2].imshow(comparison_mask, cmap=plt.cm.colors.ListedColormap(colormap))
        axes[r, 2].set_title("Prediction Comparison")
        axes[r, 2].axis("off")

    # Create a legend
    legend_patches = [
        mpatches.Patch(color=COLOR_MAPPING[1], label="Correct Vegetation (True Positive)"),
        mpatches.Patch(color=COLOR_MAPPING[2], label="False Positive"),
        mpatches.Patch(color=COLOR_MAPPING[3], label="False Negative")
    ]
    fig.legend(handles=legend_patches, loc="lower center", ncol=2, fontsize=12)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to fit legend
    # plt.show()
    result_dir = os.getcwd()
    plt.savefig(os.path.join(result_dir, f"{title}.png"))
    plt.close()

def run_evaluation(model, X_test, y_test, result_dir):
    """Run all evaluation steps."""
    print("Model Evaluation")
    print("-" * 20)

    # Prediction
    y_pred, y_pred_thresholded = predict(model, X_test)

    # Evaluate model performance
    evaluate_model(model, X_test, y_test)

    # Compute IoU
    # compute_iou(y_pred_thresholded, y_test)

    # Plot ROC curve and compute AUC
    plot_roc_curve(y_pred_thresholded, y_test, result_dir)

    # Visualize predictions vs. ground truth
    visualize_predictions(X_test, y_test, y_pred_thresholded, result_dir, title="Final Predictions")

def evaluate_model_sklearn(model, X_test, y_test):
    """Evaluate the model and return key performance metrics."""
    # Generate model predictions
    y_pred, y_pred_thresholded = predict(model, X_test)

    # Compute evaluation metrics
    loss = model.evaluate(X_test, y_test, verbose=0)[0]  # Get loss from model evaluation
    acc = accuracy_score(y_test.flatten(), y_pred_thresholded.flatten())
    precision = precision_score(y_test.flatten(), y_pred_thresholded.flatten(), zero_division=0)
    recall = recall_score(y_test.flatten(), y_pred_thresholded.flatten(), zero_division=0)
    f1 = f1_score(y_test.flatten(), y_pred_thresholded.flatten(), zero_division=0)
    auc = roc_auc_score(y_test.flatten(), y_pred.flatten())  # AUC before thresholding
    mean_iou = jaccard_score(y_test.flatten(), y_pred_thresholded.flatten(), average='micro')  # Mean IoU
    iou_class1 = jaccard_score(y_test.flatten(), y_pred_thresholded.flatten(),
                               pos_label=1, average='binary')  # IoU for vegetation (class 1)

    return loss, acc, precision, recall, f1, mean_iou, iou_class1, auc, y_pred_thresholded





