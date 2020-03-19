"""Extract weights of a classifier."""
import argparse

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras


def main(args):
    model = keras.models.load_model(args.MODEL)
    w = np.mean(model.trainable_weights[0].numpy().reshape(
        [1024, 1024, 3]), axis=2)
    w = np.abs(np.mean(model.trainable_weights[0].numpy().reshape(
        [1024, 1024, 3]), axis=2))
    plt.matshow(w, cmap=plt.cm.inferno, vmax=0.05)
    plt.colorbar(orientation="horiontal", pad=0.2)
    plt.savefig(f"weights.pdf")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "MODEL", help="Model to extract weights from.", type=str)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
