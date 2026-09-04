import numpy as np
import pandas as pd
import pytest

from src import woe_binning as wb


def test_woe_iv_from_counts_known_values():
    # bad_dist = 50/100 = 0.5, good_dist = 50/900 = 0.0556 -> woe = ln(9) approx 2.197
    woe, iv = wb.woe_iv_from_counts(bad_count=50, good_count=50, total_bad=100, total_good=900)
    assert woe == pytest.approx(np.log(9), abs=1e-3)
    assert iv == pytest.approx((0.5 - 50 / 900) * np.log(9), abs=1e-3)


def test_woe_iv_from_counts_equal_distribution_is_zero():
    # if bad_dist == good_dist, woe and iv should both be ~0
    woe, iv = wb.woe_iv_from_counts(bad_count=10, good_count=90, total_bad=100, total_good=900)
    assert woe == pytest.approx(0.0, abs=1e-6)
    assert iv == pytest.approx(0.0, abs=1e-6)


def test_calculate_woe_iv_total_iv_matches_sum_of_bins():
    bin_labels = pd.Series(["low"] * 50 + ["high"] * 50)
    target = pd.Series([1] * 40 + [0] * 10 + [0] * 45 + [1] * 5)
    result = wb.calculate_woe_iv(bin_labels, target)
    assert set(result["bin_label"]) == {"low", "high"}
    assert result["iv"].sum() > 0


def test_calculate_woe_iv_requires_both_classes():
    bin_labels = pd.Series(["a", "b", "c"])
    target = pd.Series([0, 0, 0])
    with pytest.raises(ValueError):
        wb.calculate_woe_iv(bin_labels, target)


def test_fit_and_transform_feature_woe_numeric():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=2000), name="x")
    # make target correlated with x so IV > 0
    p = 1 / (1 + np.exp(-x))
    y = pd.Series((rng.random(2000) < p).astype(int))

    fit = wb.fit_feature_woe(x, y, max_bins=5)
    assert fit["iv_total"] > 0
    assert fit["mode"] == "quantile"

    woe_values = wb.transform_feature_woe_values(x, fit)
    assert len(woe_values) == len(x)
    assert woe_values.notna().all()


def test_fit_feature_woe_low_cardinality_numeric_uses_discrete_mode():
    rng = np.random.default_rng(1)
    x = pd.Series(rng.integers(0, 2, size=1000).astype(float), name="flag")
    p = np.where(x == 1, 0.3, 0.05)
    y = pd.Series((rng.random(1000) < p).astype(int))

    fit = wb.fit_feature_woe(x, y)
    assert fit["mode"] == "discrete"
    assert fit["bin_count"] == 2
    assert fit["iv_total"] > 0


def test_select_features_by_iv_filters_range():
    iv_summary = pd.DataFrame({
        "feature_name": ["useless", "weak", "strong", "suspicious"],
        "iv_value": [0.005, 0.05, 0.3, 0.9],
        "bin_count": [3, 3, 3, 3],
    })
    selected = wb.select_features_by_iv(iv_summary, min_iv=0.02, max_iv=0.5)
    assert selected == ["strong", "weak"]
