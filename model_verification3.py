# Perform various experiments to verify the model's correctness

# Experiment 2: Input-independent baseline
# Verify that the model actually learned from the input by training a model with constant input
# and evaluate its performance

import os
import numpy as np
import tensorflow as tf
from data_preparation import read_file, split_dataset, show_statistics, inspect_dataset, formatted_print_shapes
from data_augmentation import get_data_generators, my_image_mask_generator, inspect_generator
from model_unet import unet_model
from model_evaluation_and_prediction import run_evaluation


def train_model(X_train, y_train, X_val, y_val, model_path='model.keras', zero_input=False):
    """Train a model on real data or zero-input baseline."""

    if zero_input:
        X_train = np.zeros_like(X_train)
        X_val = np.zeros_like(X_val)

    from_logits = False

    model = unet_model(
        input_shape=(128, 128, 3),
        from_logits=from_logits,
        n_filters=32,
        use_dropout=False
    )

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
                  loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=from_logits),
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(thresholds=0),
                           tf.keras.metrics.Recall(thresholds=0)
                           ])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8)
    model.save(model_path)

    return model


# --- Load dataset ---
data_dir = os.path.expanduser("~/Documents/Omdena/FrankfurtGermanyChapter_UrbanGreenSpaceMappping/MULC")
image_dir = 'VBWVA_8R'
subset_size = 120
image_dataset, mask_dataset = read_file(
    data_dir,
    image_dir,
    multiclass=False,
    if_subset=True,
    subset_size=subset_size
)

X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(image_dataset, mask_dataset)
del image_dataset, mask_dataset
formatted_print_shapes(X_train, X_val, X_test, y_train, y_val, y_test)

# --- Train Models ---

# Train normal model
if os.path.exists("model_normal.keras"):
    model_normal = tf.keras.models.load_model("model_normal.keras")  # Load trained model
else:
    print("\nTraining normal model...\n")
    model_normal = train_model(X_train, y_train, X_val, y_val, model_path="model_normal.keras")

# Train zero-input baseline model
if os.path.exists("model_zero.keras"):
    model_zero = tf.keras.models.load_model("model_zero.keras")  # Load trained model
else:
    print("\nTraining zero-input baseline model...\n")
    model_zero = train_model(X_train, y_train, X_val, y_val, model_path="model_zero.keras", zero_input=True)

# Evaluate both models
print("\nEvaluating normal model...\n")
run_evaluation(model_normal, X_val, y_val)

print("\nEvaluating zero-input baseline model...\n")
run_evaluation(model_zero, X_val, y_val)
