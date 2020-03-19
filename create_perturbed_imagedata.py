"""Script for creating perturbed image data."""
import argparse
import os

import cv2
import numpy as np
from PIL import Image

from src.dataset import image_paths


def noise(image):
    # variance from U[5.0,20.0]
    variance = np.random.uniform(low=5., high=20.)
    image = np.copy(image).astype(np.float64)
    noise = variance * np.random.randn(*image.shape)
    image += noise
    return np.clip(image, 0., 255.).astype(np.uint8)


def blur(image):
    # kernel size from [1, 3, 5, 7, 9]
    kernel_size = np.random.choice([3, 5, 7, 9])
    return cv2.GaussianBlur(
        image, (kernel_size, kernel_size), sigmaX=cv2.BORDER_DEFAULT)


def jpeg(image):
    # qualite factor sampled from U[10, 75]
    factor = np.random.randint(low=10, high=75)
    _, image = cv2.imencode(".jpg", image, [factor, 90])
    return cv2.imdecode(image, 1)


def cropping(image):
    # crop between 5% and 20%
    percentage = np.random.uniform(low=.05, high=.2)
    x, y, _ = image.shape
    x_crop = int(x * percentage * .5)
    y_crop = int(y * percentage * .5)
    cropped = image[x_crop:-x_crop, y_crop:-y_crop]
    resized = cv2.resize(cropped, dsize=(x, y), interpolation=cv2.INTER_CUBIC)
    return resized


def apply_transformation_to_datasets(datasets, mode, size):
    if mode == "noise":
        image_functions = [noise]

    elif mode == "blur":
        image_functions = [blur]

    elif mode == "jpeg":
        image_functions = [jpeg]

    elif mode == "cropping":
        image_functions = [cropping]

    elif mode == "combined":
        image_functions = [noise, blur, jpeg, cropping]

    else:
        raise NotImplementedError("Selected unrecognized mode: {mode}!")

    for dir_path in datasets:
        output_dir = f"{dir_path}_{mode}"
        os.makedirs(output_dir, exist_ok=True)
        paths = image_paths(dir_path)[:size]
        images = map(np.asarray, map(Image.open, paths))

        for i, image in enumerate(images):
            current_function = image_functions.pop(0)

            new_image = image
            if np.random.sample() > .5:
                new_image = current_function(new_image)
                assert not np.isclose(new_image, image).all()

            Image.fromarray(new_image).save(f"{output_dir}/{mode}_{i:06}.png")
            image_functions.append(current_function)
            print(
                f"\rConverted {i+1: 6} out of {len(paths) if size is None else max(len(paths), size)} images for {dir_path}!", end="")

        print(f"\nFinished converting {dir_path}!")


def main(args):
    if args.MODE == "all":
        apply_transformation_to_datasets(args.DATASETS, "noise", args.size)
        apply_transformation_to_datasets(args.DATASETS, "blur", args.size)
        apply_transformation_to_datasets(args.DATASETS, "cropping", args.size)
        apply_transformation_to_datasets(args.DATASETS, "jpeg", args.size)
        apply_transformation_to_datasets(args.DATASETS, "combined", args.size)
    else:
        apply_transformation_to_datasets(args.DATASETS, args.MODE, args.size)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "MODE", help="Mode: {noise, blur, cropping, jpeg, combined, all}.", type=str)
    parser.add_argument(
        "DATASETS", help="Datasets to modify.", type=str, nargs="+")
    parser.add_argument(
        "--size", "-s", help="Amount of datafiles to use.", type=int, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
