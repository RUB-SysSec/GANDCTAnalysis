import numpy as np
from src.math import welford


def test_welford():
    for _ in range(1_000):
        random_data = np.random.randn(5, 5, 1_000)
        mean, var = welford(random_data)

        assert np.isclose(random_data.mean(axis=0), mean).all()
        assert np.isclose(random_data.var(axis=0), var).all()
