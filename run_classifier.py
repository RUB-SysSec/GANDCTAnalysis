import argparse
import functools
import multiprocessing
from collections import Counter

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from src.dataset import image_paths
from src.image import dct2
from src.image_np import load_image


class DCTLayer(tf.keras.layers.Layer):
    def __init__(self, mean, var):
        super(DCTLayer, self).__init__()
        self.mean = mean
        self.var = var
        self.std = np.sqrt(var)

    def build(self, input_shape):
        self.mean_w = self.add_weight(
            "mean", shape=input_shape[1:], initializer=tf.keras.initializers.Constant(self.mean), trainable=False)
        self.std_w = self.add_weight(
            "std", shape=input_shape[1:], initializer=tf.keras.initializers.Constant(self.std), trainable=False)

    def call(self, inputs):
        # dct2
        x = dct2(inputs)

        # log scale
        x = tf.abs(x)
        x += 1e-13
        x = tf.math.log(x)

        # remove mean + unit variance
        x = x - self.mean_w
        x = x / self.std_w

        return x


class PixelLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(PixelLayer, self).__init__()

    def call(self, inputs):
        x = (inputs / 127.5) - 1.
        return x


def _pixel(model):
    return tf.keras.Sequential([PixelLayer(), model])


def _dct(model, mean, var):
    return tf.keras.Sequential([DCTLayer(mean, var), model])


def _process_image(path):
    image = load_image(path)
    image = image.astype(np.float32)
    return image


def _load_images(path, amount=None):
    paths = image_paths(path)
    if amount is not None:
        paths = paths[:amount]

    images = multiprocessing.Pool(
        multiprocessing.cpu_count()).map(_process_image, paths)
    return images


def main(args):
    model = tf.keras.models.load_model(args.MODEL)
    model.summary()

    if args.dct is not None:
        model = _dct(model, np.load(f"{args.dct}/mean.npy"),
                     np.load(f"{args.dct}/var.npy"))
    else:
        model = _pixel(model)

    images = _load_images(args.DATA, amount=args.size)
    class_counts = Counter()
    for i in tqdm(range(0, len(images), args.batch_size)):
        predictions = model(np.asarray(images[i:i+args.batch_size]))

        if predictions.shape[1] > 1:
            predictions = tf.argmax(predictions, axis=1)
        else:
            predictions = tf.math.round(predictions)
            predictions = tf.cast(predictions, tf.uint8)
            predictions = tf.reshape(predictions, shape=(-1,))

        for pred in predictions:
            class_counts[pred.numpy()] += 1

    for c, amount in sorted(class_counts.items(), key=lambda x: x[0]):
        print(f"{amount/len(images): 3.2%} of the images are from class {c} ({amount})")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("MODEL", help="Model to evaluate.", type=str)
    parser.add_argument("DATA", help="Directory to classify.")

    parser.add_argument(
        "--size", "-s", help="Only use this amount of images.", type=int, default=None)

    batch_size = 32
    parser.add_argument(
        "--batch_size", "-b", help="Batch size to use; Default: {batch_size}.", type=int, default=batch_size)
    parser.add_argument(
        "--dct", "-d", help="DCT input", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
