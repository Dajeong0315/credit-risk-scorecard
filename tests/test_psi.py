import numpy as np
import pandas as pd

from src import psi


def test_calculate_psi_identical_distributions_is_near_zero():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=5000))
    value = psi.calculate_psi(x, x, bins=10)
    assert value == 0.0 or abs(value) < 1e-6


def test_calculate_psi_shifted_distribution_is_positive_and_larger():
    rng = np.random.default_rng(1)
    expected = pd.Series(rng.normal(loc=0, scale=1, size=5000))
    slightly_shifted = pd.Series(rng.normal(loc=0.1, scale=1, size=5000))
    heavily_shifted = pd.Series(rng.normal(loc=1.5, scale=1, size=5000))

    small_psi = psi.calculate_psi(expected, slightly_shifted, bins=10)
    large_psi = psi.calculate_psi(expected, heavily_shifted, bins=10)

    assert small_psi >= 0
    assert large_psi > small_psi
