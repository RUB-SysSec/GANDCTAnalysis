import numpy as np

from src.image import dct2
from src.image_np import dct2 as dct2_np


def test_dct():
    for i in range(1_000):
        data = np.random.randn(5, 5, 1).astype(np.float32)

        tf_dct = dct2(data, batched=False).numpy()
        np_dct = dct2_np(data)

        # smaller absolute differnce since we compare values smaller than 0
        # see https://docs.scipy.org/doc/numpy/reference/generated/numpy.isclose.html#numpy.isclose
        assert np.allclose(tf_dct, np_dct, atol=1e-06)
