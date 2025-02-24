# Perform various experiments to verify the model's correctness

# Experiment 1: Verify model loss at initialization
# Loss = binary focal loss
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryFocalCrossentropy

from model_unet import unet_model


# Create dummy data
dummy_input = np.random.rand(1, 128, 128, 3).astype(np.float32) #Single image
dummy_mask = np.full((1, 128, 128, 1), 1.0, dtype=np.float32) #Single mask filled with 1.0

# Initialize the model
model = unet_model(input_shape=(128, 128, 3), from_logits=False, n_filters=64, use_dropout=False)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=BinaryFocalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# Calculate the initial loss
initial_loss = model.evaluate(dummy_input, dummy_mask, verbose=0)
print(f"Initial Loss: {initial_loss[0]:.4f}")

# Calculate manually for verification
predictions = model.predict(dummy_input)
loss_fn = BinaryFocalCrossentropy(from_logits=False)
manual_loss = loss_fn(dummy_mask, predictions).numpy()
print(f"Manual Loss: {manual_loss:.4f}")

expected_loss = -0.25 * 0.5**2 * np.log(0.5)
print(f"Expected Loss: {expected_loss:.4f}")

# Plot distribution of predictions
import matplotlib.pyplot as plt

plt.hist(predictions.flatten(), bins=50)
plt.title("Predictions Histogram")
plt.show()
print(f"Predictions Mean: {predictions.mean():.4f}")
print(f"Predictions Std Dev: {predictions.std():.4f}")

# Result: Didn't expected loss value
# loss = binary focal loss
# Expected initial loss = -0.25 * 0.5**2 * log(0.5)

# n_filters = 32, Initial Loss: 0.1700, Predictions Mean: 0.5029
# n_filters = 64, Initial Loss: 0.1825, Predictions Mean: 0.4925
# There's quite some variation in initial loss

