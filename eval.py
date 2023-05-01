import os
import argparse
import numpy as np
from keras.models import load_model
from PIL import Image
import tensorflow as  tf
from model import get_model

IMAGE_SIZE = 128

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

def load_data(low_light_image_path):
    low_light_image = read_image(low_light_image_path)
    return low_light_image

def evaluate_model(images_path, eval_path, model_path):
    model = get_model()
    model.load_weights("./model.h5")

    image_files = [f for f in os.listdir(images_path) if f.endswith(".jpg") or f.endswith(".png")]

    for image_file in image_files:
        image_path = os.path.join(images_path, image_file)
        image = load_data(image_path)
        image = np.expand_dims(image, axis=0)

        generated_image = model.predict(image)

        eval_file = f"eval_{image_file}"
        eval_file_path = os.path.join(eval_path, eval_file)
        generated_image = np.squeeze(generated_image, axis=0)
        generated_image = np.clip(generated_image * 255, 0, 255).astype(np.uint8)  # Clip values and convert to uint8
        generated_image = Image.fromarray(generated_image)
        generated_image.save(eval_file_path)
        print(f"Generated image saved at: {eval_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate model on images")
    parser.add_argument("images_path", type=str, help="Path to images directory")
    parser.add_argument("eval_path", type=str, help="Path to evaluation directory")
    parser.add_argument("--model_path", type=str, default="./model.h5", help="Path to model")
    args = parser.parse_args()
    os.makedirs(args.eval_path, exist_ok=True)
    evaluate_model(args.images_path, args.eval_path, args.model_path)

if __name__ == "__main__":
    main()