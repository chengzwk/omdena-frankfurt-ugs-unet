# End-to-end U-Net model training

# Import packages
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from focal_loss import BinaryFocalLoss
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from keras.callbacks import EarlyStopping

# Import functions from other scripts
from data_preparation import read_file, split_dataset, show_statistics, inspect_dataset
from data_augmentation import get_data_generators, my_image_mask_generator, inspect_generator
from model_unet import unet_model


def plot_accuracy(history_2):
    loss = history_2.history['loss']
    val_loss = history_2.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    acc = history_2.history['accuracy']
    val_acc = history_2.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training acc')
    plt.plot(epochs, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Load dataset
data_dir = os.path.expanduser("~/Documents/Omdena/FrankfurtGermanyChapter_UrbanGreenSpaceMappping/MULC")
image_dir = 'VBWVA_8R'
image_dataset, mask_dataset = read_file(image_dir, multiclass=False, if_subset=True, subset_size=100)

# Print statistics of the dataset and visually inspect the dataset
show_statistics(image_dataset, mask_dataset)
inspect_dataset(image_dataset, mask_dataset)

# Train-validation-test split
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(image_dataset, mask_dataset)
del image_dataset, mask_dataset
print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)

# Data augmentation
batch_size = 16
steps_per_epoch = len(X_train)//batch_size  # for generator
validation_steps = len(X_val)//batch_size  # for generator
print(steps_per_epoch, validation_steps)
image_generator, valid_img_generator, mask_generator, valid_mask_generator = \
    get_data_generators(X_train, X_val, y_train, y_val, batch_size=batch_size)

# Inspect data generators
inspect_generator(image_generator, mask_generator)
inspect_generator(valid_img_generator, valid_mask_generator)

# Combine image-mask generators
train_generator = my_image_mask_generator(image_generator, mask_generator)
validation_generator = my_image_mask_generator(valid_img_generator, valid_mask_generator)

# Build model
model = unet_model(input_shape=(128, 128, 3), use_dropout=False)

# Compile model
model.compile(optimizer = Adam(learning_rate = 1e-4),
              loss = BinaryFocalLoss(gamma = 2),
              metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Train model
epochs = 50  # Set epochs and early stopping

history_2 = model.fit(
    train_generator,
    validation_data=validation_generator,
    batch_size=batch_size,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    epochs = epochs
)

# save model
model.save('unet.h5')

# Plot the training and validation accuracy and loss at each epoch
plot_accuracy(history_2)



