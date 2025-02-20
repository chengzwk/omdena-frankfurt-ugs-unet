# Data augmentation was performed to expand image dataset and to reduce overfitting.
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator


def get_data_generators(X_train, X_val, y_train, y_val, use_augmentation=True, batch_size=16, seed=24):
    """Create ImageDataGenerator for images and masks."""
    if use_augmentation:
        # Set parameters of data augmentation
        img_data_gen_args = dict(rotation_range=45,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 fill_mode='reflect')

        mask_data_gen_args = dict(rotation_range=45,
                                  width_shift_range=0.2,
                                  height_shift_range=0.2,
                                  zoom_range=0.2,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  fill_mode='reflect',
                                  preprocessing_function = lambda x: np.where(x > 0, 1, 0).astype(x.dtype))

        # image generator (on X_train and X_val)
        image_data_generator = ImageDataGenerator(**img_data_gen_args)
        image_data_generator.fit(X_train, augment=True, seed=seed)  # load X_train before this

        # mask generator (on y_train and y_val)
        mask_data_generator = ImageDataGenerator(**mask_data_gen_args)
        mask_data_generator.fit(y_train, augment=True, seed=seed)  # load y_train before this

    else:
        # No data augmentation
        image_data_generator = ImageDataGenerator()
        image_data_generator.fit(X_train, seed=seed)

        mask_data_generator = ImageDataGenerator()
        mask_data_generator.fit(y_train, seed=seed)

    image_generator = image_data_generator.flow(X_train, seed=seed, batch_size=batch_size)
    valid_img_generator = image_data_generator.flow(X_val, seed=seed, batch_size=batch_size)

    mask_generator = mask_data_generator.flow(y_train, seed=seed, batch_size=batch_size)
    valid_mask_generator = mask_data_generator.flow(y_val, seed=seed, batch_size=batch_size)

    return image_generator, valid_img_generator, mask_generator, valid_mask_generator

def my_image_mask_generator(image_generator, mask_generator):
    """Put image generator and mask generator together"""
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

def inspect_generator(image_generator, mask_generator):
    x = image_generator.next()
    y = mask_generator.next()

    for i in range(0, 1):
        plt.figure(figsize=(8, 4))
        image = x[i]
        mask = y[i]
        plt.subplot(1, 2, 1)
        plt.imshow(image[:, :, ])

        plt.subplot(1, 2, 2)
        plt.imshow(mask[:, :, 0], cmap='gray')
        plt.show()


"""
train_generator = my_image_mask_generator(image_generator, mask_generator)
validation_generator = my_image_mask_generator(valid_img_generator, valid_mask_generator)
# train_generator and validation_generator can be directly used in model.fit()
"""



