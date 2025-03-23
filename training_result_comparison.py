import os
import numpy as np
import pickle
import matplotlib.pyplot as plt


def plot_comparison(history_before, history_after, train_metric, val_metric, title, \
                    train_metric_before=None, val_metric_before=None):
    if train_metric_before is not None:
        epochs_before = range(1, len(history_before[train_metric_before]) + 1)
    else:
        epochs_before = range(1, len(history_before[train_metric]) + 1)
    epochs_after = range(1, len(history_after[train_metric]) + 1)

    plt.figure(figsize=(10, 5))
    if train_metric_before is not None:
        plt.plot(epochs_before, history_before[train_metric_before], 'blue', label=f'Training/Before')
        plt.plot(epochs_before, history_before[val_metric_before], 'lightblue', label=f'Validation/Before')
    else:
        plt.plot(epochs_before, history_before[train_metric], 'blue', label=f'Training/Before')
        plt.plot(epochs_before, history_before[val_metric], 'lightblue', label=f'Validation/Before')

    plt.plot(epochs_after, history_after[train_metric], 'red', label=f'Training/After')
    plt.plot(epochs_after, history_after[val_metric], 'lightcoral', label=f'Validation/After')
    plt.xlabel('Epochs')
    plt.ylabel(train_metric)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{title}.png")

# Set paths
before_path = "experiment08_decrease_learning_rate/unet_training_history.pkl"
after_path = "experiment01_add_stlouis_data/unet_training_history.pkl"

# Load histories
def load_history(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)

history_before = load_history(before_path)
history_after = load_history(after_path)

# Compare Loss
plot_comparison(history_before, history_after, "loss", "val_loss","Loss Comparison")

# Compare Accuracy
plot_comparison(history_before, history_after, "accuracy", "val_accuracy", "Accuracy Comparison")

# Compare IoU
plot_comparison(history_before, history_after, "binary_io_u", "val_binary_io_u", \
                "Mean IoU Comparison", train_metric_before="binary_io_u_8", val_metric_before="val_binary_io_u_8")

# Assuming you manually recorded the final evaluation metrics:
metrics_before = {
    "Loss": 0.4156,
    "Accuracy": 0.7739,
    "Precision": 0.8068,
    "Recall": 0.8280,
    "F1 Score": 0.8173,
    "Mean IoU": 0.6129,
    "IoU for class 1": 0.6910,
    "AUC": 0.7552
}

metrics_after = {
    "Loss": 0.1667,
    "Accuracy": 0.7811,
    "Precision": 0.8528,
    "Recall": 0.8108,
    "F1 Score": 0.8313,
    "Mean IoU": 0.6533,
    "IoU for class 1": 0.7113,
    "AUC": 0.7933
}

# Compare in a table
print("\nComparison of Key Metrics:")
print(f"{'Metric':<15} {'Before':<10} {'After':<10} {'Change':<10}")
for key in metrics_before:
    change = metrics_after[key] - metrics_before[key]
    print(f"{key:<15} {metrics_before[key]:<10.4f} {metrics_after[key]:<10.4f} {change:<10.4f}")


from PIL import Image

roc_path_before = "experiment08_decrease_learning_rate/ROC_curve.png"
roc_path_after = "experiment01_add_stlouis_data/ROC_curve.png"

def show_image(image_path, title):
    img = Image.open(image_path)
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()

show_image(roc_path_before, "ROC Curve Before Adding Data")
show_image(roc_path_after, "ROC Curve After Adding Data")
