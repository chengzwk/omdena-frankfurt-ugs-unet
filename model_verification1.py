# Perform various experiments to verify the model's correctness

# Experiment 1: Verify model loss at initialization
# Loss = binary cross entropy loss
import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

from model_unet import unet_model


# Create dummy data
dummy_input = np.random.rand(1, 128, 128, 3).astype(np.float32) #Single image
dummy_mask = np.full((1, 128, 128, 1), 0.5, dtype=np.float32) #Single mask filled with 0.5s

# Initialize the model
model = unet_model(input_shape=(128, 128, 3), from_logits=True, n_filters=64, use_dropout=False)

# Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss=BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Calculate the initial loss
initial_loss = model.evaluate(dummy_input, dummy_mask, verbose=0)
print(f"Initial Loss: {initial_loss[0]:.4f}")

# Calculate manually for verification
predictions = model.predict(dummy_input)
loss_fn = BinaryCrossentropy(from_logits=True)
manual_loss = loss_fn(dummy_mask, predictions).numpy()
print(f"Manual Loss: {manual_loss:.4f}")

expected_loss = np.log(2)
print(f"Expected Loss: {expected_loss:.4f}")

# Result: Got expected loss value
# n_filters = 32
# n_filters = 64
# loss = binary cross entropy loss
# Expected initial loss = -log(0.5)

