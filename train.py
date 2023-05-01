import os
import random
import numpy as np
from glob import glob
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from model import get_model

# functions to create the dataset
random.seed(1)
IMAGE_SIZE = 128
BATCH_SIZE = 4
MAX_TRAIN_IMAGES = 300

def autocontrast(tensor, cutoff=0):
    tensor = tf.cast(tensor, dtype=tf.float32)
    min_val = tf.reduce_min(tensor)
    max_val = tf.reduce_max(tensor)
    range_val = max_val - min_val
    adjusted_tensor = tf.clip_by_value(tf.cast(tf.round((tensor - min_val - cutoff) * (255 / (range_val - 2 * cutoff))), tf.uint8), 0, 255)
    return adjusted_tensor

def read_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = autocontrast(image)
    image.set_shape([None, None, 3])
    image = tf.cast(image, dtype=tf.float32) / 255
    return image


def random_crop(low_image, enhanced_image):
    low_image_shape = tf.shape(low_image)[:2]
    low_w = tf.random.uniform(
        shape=(), maxval=low_image_shape[1] - IMAGE_SIZE + 1, dtype=tf.int32
    )
    low_h = tf.random.uniform(
        shape=(), maxval=low_image_shape[0] - IMAGE_SIZE + 1, dtype=tf.int32
    )
    enhanced_w = low_w
    enhanced_h = low_h
    low_image_cropped = low_image[
        low_h : low_h + IMAGE_SIZE, low_w : low_w + IMAGE_SIZE
    ]
    enhanced_image_cropped = enhanced_image[
        enhanced_h : enhanced_h + IMAGE_SIZE, enhanced_w : enhanced_w + IMAGE_SIZE
    ]
    return low_image_cropped, enhanced_image_cropped


def load_data(low_light_image_path, enhanced_image_path):
    low_light_image = read_image(low_light_image_path)
    enhanced_image = read_image(enhanced_image_path)
    low_light_image, enhanced_image = random_crop(low_light_image, enhanced_image)
    return low_light_image, enhanced_image


def get_dataset(low_light_images, enhanced_images):
    dataset = tf.data.Dataset.from_tensor_slices((low_light_images, enhanced_images))
    dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    return dataset

# Loss functions

class CustomLoss:
    def __init__(self, perceptual_loss_model):
        self.perceptual_loss_model = perceptual_loss_model
    def perceptual_loss(self, y_true, y_pred):
        y_true_features = self.perceptual_loss_model(y_true)
        y_pred_features = self.perceptual_loss_model(y_pred)
        loss = tf.reduce_mean(tf.square(y_true_features[0] - y_pred_features[0])) + tf.reduce_mean(tf.square(y_true_features[1] - y_pred_features[1]))
        return loss
    def charbonnier_loss(self, y_true, y_pred):
        return tf.reduce_mean(tf.sqrt(tf.square(y_true - y_pred) + tf.square(1e-3)))
    def __call__(self, y_true, y_pred):
        return  0.5*self.perceptual_loss(y_true, y_pred)  + 0.4*self.charbonnier_loss(y_true, y_pred)

def peak_signal_noise_ratio(y_true, y_pred):
    return tf.image.psnr(y_pred, y_true, max_val=255.0)

def main():
    train_low_light_images = sorted(glob("./lol_dataset/our485/low/*"))[:MAX_TRAIN_IMAGES]
    train_enhanced_images = sorted(glob("./lol_dataset/our485/high/*"))[:MAX_TRAIN_IMAGES]

    val_low_light_images = sorted(glob("./lol_dataset/our485/low/*"))[MAX_TRAIN_IMAGES:]
    val_enhanced_images = sorted(glob("./lol_dataset/our485/high/*"))[MAX_TRAIN_IMAGES:]

    train_dataset = get_dataset(train_low_light_images, train_enhanced_images)
    val_dataset = get_dataset(val_low_light_images, val_enhanced_images)

    #Model for calculating perceptual loss
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    for layer in vgg.layers:
        layer.trainable = False  #Freeze all the layers, since this model is for evaluation only
    outputs = [vgg.get_layer('block3_conv3').output, vgg.get_layer('block4_conv3').output]
    perceptual_loss_model = tf.keras.models.Model(inputs=vgg.input, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss = CustomLoss(perceptual_loss_model)
    model = get_model()

    model.compile(
        optimizer=optimizer, loss=loss, metrics=[peak_signal_noise_ratio]
    )

    history = model.fit(train_dataset, validation_data=val_dataset, epochs=50)
    model.save_weights("model.h5")

if __name__ == "__main__":
    main()