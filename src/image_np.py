import numpy as np
from PIL import Image
from scipy import fftpack


def dct2(array):
    array = fftpack.dct(array, type=2, norm="ortho", axis=0)
    array = fftpack.dct(array, type=2, norm="ortho", axis=1)
    return array


def fft2d(array):
    array = fftpack.fft2(array)
    array = fftpack.fftshift(array)
    return array


def load_image(path, grayscale=False, tf=False):
    x = Image.open(path)
    if grayscale:
        x = x.convert("L")
        if tf:
            x = np.asarray(x)
            x = np.reshape(x, [*x.shape, 1])
    return np.asarray(x)


def normalize(image, mean, std):
    image = (image - mean) / std
    return image


def normalize_with_label(inputs, mean, std):
    image, label = inputs
    image = normalize_with_label(image, mean, std)
    return image, label


def scale_image(image):
    if not image.flags.writeable:
        image = np.copy(image)

    if image.dtype == np.uint8:
        image = image.astype(np.float32)
    image /= 127.5
    image -= 1.
    return image


def scale_image_with_lable(inputs):
    image, label = inputs
    image = scale_image(image)
    return (image, label)
