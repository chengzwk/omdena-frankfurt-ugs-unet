# Perform various experiments to verify the model's correctness

# Experiment 4: Overfit a single batch
# Overfit a single batch of 2 examples and verify we can reach the lowest achievable loss

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from markdown_it.rules_inline import image

from data_preparation import read_file, split_dataset
from data_augmentation import my_image_mask_generator
from model_unet import unet_model
from model_evaluation_and_prediction import run_evaluation

# Track experiment with MLflow
import dagshub
import mlflow
dagshub.init(repo_name="omdena-frankfurt-ugs-unet", repo_owner="chengzwk")
mlflow.tensorflow.autolog()


# ==== Step 1: Load & Preprocess Small Batch ====
def load_small_batch(data_dir, image_dir, batch_size):
    """Load a small batch of data for overfitting test."""
    X_train, y_train = read_file(
        data_dir,
        image_dir,
        multiclass=False,
        subset_size=batch_size
    )
    return X_train, y_train


# ==== Step 2: Modify Model for High Capacity ====
def get_high_capacity_unet():
    """Create a U-Net model with increased complexity to encourage overfitting."""
    model = unet_model(input_shape=(128, 128, 3), from_logits=True, n_filters=32, use_dropout=False)  # Increase filters
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
        # loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=True),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model


# ==== Step 3: Train Until Overfitting ====
def train_on_small_batch(model, X_train, y_train, epochs=1000):
    """Train the model on a single batch until it memorizes the data."""
    print("\nStarting Overfit Training on One Batch...")

    history = model.fit(
        X_train, y_train,
        batch_size=len(X_train),  # Fit on full batch at once
        epochs=epochs,
        verbose=1
    )

    return history


# ==== Step 4: Visualize Predictions ====
def visualize_predictions(model, X_train, y_train, threshold=0.5):
    """Plot ground truth vs predicted masks."""
    # y_pred = model.predict(X_train)
    y_pred = model(X_train, training=True)
    y_pred = tf.sigmoid(y_pred).numpy()
    y_pred_thresholded = (y_pred >= threshold).astype(int)

    fig, axes = plt.subplots(len(X_train), 4, figsize=(10, 5 * len(X_train)))

    for i in range(len(X_train)):
        axes[i, 0].imshow(X_train[i].squeeze())
        axes[i, 0].set_title("Input Image")

        axes[i, 1].imshow(y_train[i].squeeze(), cmap="gray")
        axes[i, 1].set_title("Ground Truth")

        axes[i, 2].imshow(y_pred[i].squeeze(), cmap="gray")
        axes[i, 2].set_title("Predicted Mask")

        axes[i, 3].imshow(y_pred_thresholded[i].squeeze(), cmap="gray")
        axes[i, 3].set_title("Predicted Mask after threshold")

    plt.show()


# ==== Step 5: Evaluate Final Performance ====
def run_overfit_experiment(data_dir, image_dir, batch_size, epochs, model_name):
    # Get a high-capacity model
    model = get_high_capacity_unet()

    # Train until overfitting
    with mlflow.start_run():
        train_on_small_batch(model, X_train, y_train, epochs=epochs)

    # Save model
    model.save(model_name)
    return model


# ==== Run Experiment ====
if __name__ == "__main__":
    data_dir = os.path.expanduser("~/Documents/Omdena/FrankfurtGermanyChapter_UrbanGreenSpaceMappping/MULC")
    image_dir = 'VBWVA_8R'
    batch_size = 2
    epochs = 200
    X_train, y_train = load_small_batch(data_dir, image_dir, batch_size=batch_size)  # Load small batch
    model_name = 'unet_smallbatch.keras'

    if not os.path.exists(model_name):
        model = run_overfit_experiment(data_dir, image_dir, batch_size, epochs, model_name)
    else:
        model = tf.keras.models.load_model(model_name)

    visualize_predictions(model, X_train, y_train, threshold=0.5)
    run_evaluation(model, X_train, y_train)


# Result: Model reached near-zero loss on 2 training images
# loss = binary cross entropy loss
# accuracy: 1.0000 - loss: 0.0091 - precision: 1.0000 - recall: 1.0000
# Visualization shows predicted mask aligns perfectly with ground truth
# loss = binary focal loss
# accuracy: 1.0000 - loss: 0.0066 - precision: 1.0000 - recall: 1.0000
# Visualization shows predicted mask aligns perfectly with ground truth

