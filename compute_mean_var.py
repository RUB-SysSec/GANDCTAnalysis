import argparse
import os

import numpy as np

from src.dataset import image_paths
from src.image_np import dct2, load_image
from src.math import log_scale, welford


def main(args):
    paths = list()
    for data in args.DATASETS:
        paths += image_paths(data)[:args.AMOUNT]

    images = map(load_image, paths)
    images = map(dct2, images)

    mean, var = welford(images)

    os.makedirs(args.output, exist_ok=True)
    np.save(open(f"{args.output}/mean.npy", "wb+"), mean)
    np.save(open(f"{args.output}/var.npy", "wb+"), var)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "AMOUNT", help="Amount of images to use", type=int)
    parser.add_argument(
        "DATASETS", help="Directories over which to compute.", type=str, nargs="+")
    output = "saved_mean_var"
    parser.add_argument(
        "--output", "-o", help=f"Output direcotory. Default: {output}", default=output)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
