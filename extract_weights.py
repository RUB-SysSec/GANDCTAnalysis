"""Extract weights of a classifier."""
import argparse

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras

WEIGHT_SHAPE = [256, 256, 3]


def main(args):
    model = keras.models.load_model(args.MODEL)
    w = np.mean(model.trainable_weights[0].numpy().reshape(
        WEIGHT_SHAPE), axis=2)
    w = np.abs(np.mean(model.trainable_weights[0].numpy().reshape(
        WEIGHT_SHAPE), axis=2))
    plt.matshow(w, cmap=plt.cm.inferno, vmax=0.04)
    plt.colorbar(pad=0.2)
    plt.savefig(f"{args.output}.pdf")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "MODEL", help="Model to extract weights from.", type=str)
    parser.add_argument(
        "--output", "-o", help="Output file name.", type=str, default="weights")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
