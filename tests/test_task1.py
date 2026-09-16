"""A small, targeted Task 1 unit-test suite.

Not a giant test suite -- deliberately limited to the handful of design
claims where a subtle bug would be genuinely dangerous and easy to miss by
inspection: that the leakage-safe feature transformer really doesn't learn
anything from data passed to `.transform()` (only from `.fit()`), that the
group-aware split really has zero overlapping house ids between dev and
test, that the Monte Carlo p-value correction really can never return an
unqualified zero, and that the paired A/B metric functions are actually
being fed paired, equal-length prediction arrays (a length mismatch would
silently misalign predictions to the wrong houses).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import HouseFeatureEngineer
from evaluation import (
    group_train_test_split, monte_carlo_p_value, compute_metrics,
    paired_bootstrap_ci, paired_permutation_test, rmse,
)


def _toy_house_df(n=200, seed=0):
    rng = np.random.RandomState(seed)
    n_houses = n // 2  # deliberately create repeat-sale ids, like the real dataset
    ids = rng.randint(0, n_houses, size=n)
    return pd.DataFrame({
        "id": ids,
        "date": ["20140601T000000"] * n,
        "bedrooms": rng.randint(1, 6, size=n).astype(float),
        "bathrooms": rng.uniform(1, 4, size=n),
        "sqft_living": rng.uniform(500, 5000, size=n),
        "sqft_lot": rng.uniform(1000, 20000, size=n),
        "yr_built": rng.randint(1950, 2015, size=n),
        "yr_renovated": np.zeros(n),
        "sqft_basement": np.zeros(n),
        "price": rng.uniform(100_000, 900_000, size=n),
    })


# ---------------------------------------------------------------------------
# 1. Leakage: the feature engineer must not learn anything from transform-time data.
# ---------------------------------------------------------------------------

def test_feature_engineer_does_not_learn_from_transform_data():
    """Fit on data with a LOW bedroom cap and tight ratio medians, then
    transform data with WILDLY different bedroom/ratio values. If the
    transformer were (incorrectly) re-deriving its constants from the data
    being transformed -- the exact bug the earlier version of this pipeline
    had -- the clip/imputation behavior on the transform-time data would
    change when the fit-time data changes. It must not."""
    train_low = _toy_house_df(n=200, seed=1)
    train_low["bedrooms"] = 2.0  # every fit-time house has exactly 2 bedrooms

    train_high = _toy_house_df(n=200, seed=1)
    train_high["bedrooms"] = 10.0  # a very different fit-time population

    # Same transform-time data in both cases, with a deliberately extreme
    # bedroom value that would be clipped very differently depending on
    # which population the cap was learned from.
    transform_data = _toy_house_df(n=5, seed=2)
    transform_data["bedrooms"] = 999.0

    engineer_a = HouseFeatureEngineer().fit(train_low)
    engineer_b = HouseFeatureEngineer().fit(train_high)

    assert engineer_a.bedroom_cap_ != engineer_b.bedroom_cap_, (
        "sanity check: the two fits should have learned different caps"
    )

    out_a = engineer_a.transform(transform_data)
    out_b = engineer_b.transform(transform_data)

    # The transformed bedroom values must reflect EACH transformer's OWN
    # fit-time cap, not anything derived from transform_data itself (which
    # is identical in both calls).
    assert (out_a["bedrooms"] == engineer_a.bedroom_cap_).all()
    assert (out_b["bedrooms"] == engineer_b.bedroom_cap_).all()
    assert not out_a["bedrooms"].equals(out_b["bedrooms"]), (
        "transform output should differ because it depends on which fit "
        "produced the transformer, not on the (identical) transform-time data"
    )


def test_feature_engineer_transform_is_deterministic_and_stateless_per_call():
    """Calling .transform() twice on the same data, from the same fitted
    transformer, must give identical results -- transform must not mutate
    or accumulate state from repeated calls."""
    train = _toy_house_df(n=150, seed=3)
    test = _toy_house_df(n=50, seed=4)
    engineer = HouseFeatureEngineer().fit(train)
    out1 = engineer.transform(test)
    out2 = engineer.transform(test)
    pd.testing.assert_frame_equal(out1, out2)


# ---------------------------------------------------------------------------
# 2. Group split: zero house-id overlap between dev and test.
# ---------------------------------------------------------------------------

def test_group_split_has_no_id_overlap():
    df = _toy_house_df(n=400, seed=5)
    dev, test = group_train_test_split(df, group_col="id", test_frac=0.3, seed=42)
    assert set(dev["id"]).isdisjoint(set(test["id"]))
    # every row must land in exactly one of the two sets
    assert len(dev) + len(test) == len(df)


def test_group_split_is_deterministic_given_same_seed():
    df = _toy_house_df(n=400, seed=6)
    dev1, test1 = group_train_test_split(df, group_col="id", test_frac=0.3, seed=7)
    dev2, test2 = group_train_test_split(df, group_col="id", test_frac=0.3, seed=7)
    assert set(dev1["id"]) == set(dev2["id"])
    assert set(test1["id"]) == set(test2["id"])


# ---------------------------------------------------------------------------
# 3. Monte Carlo p-value correction never returns an unqualified zero.
# ---------------------------------------------------------------------------

def test_monte_carlo_p_value_never_zero():
    p = monte_carlo_p_value(extreme_count=0, n_resamples=5000)
    assert p == 1 / 5001
    assert p > 0


def test_monte_carlo_p_value_at_extremes():
    # Every resample as extreme as the observed value -> p should be 1.0,
    # the correction's other boundary case.
    p_all = monte_carlo_p_value(extreme_count=5000, n_resamples=5000)
    assert p_all == 1.0
    # Roughly half extreme -> p should sit near 0.5, not be distorted by
    # the +1/+1 correction into something misleading at moderate counts.
    p_half = monte_carlo_p_value(extreme_count=2500, n_resamples=5000)
    assert abs(p_half - 0.5) < 0.01


# ---------------------------------------------------------------------------
# 4. Paired A/B metric functions require paired, equal-length predictions.
# ---------------------------------------------------------------------------

def test_ab_metrics_reject_mismatched_lengths():
    """compute_metrics and the paired bootstrap/permutation entry points
    all subtract prediction arrays elementwise. A silent length mismatch
    would misalign predictions to the wrong houses instead of failing
    loudly -- confirm numpy actually raises rather than broadcasting
    something misleading."""
    y_true = np.array([100.0, 200.0, 300.0])
    pred_a = np.array([110.0, 190.0, 310.0])
    pred_b_wrong_length = np.array([105.0, 195.0])  # deliberately misaligned

    with pytest.raises(ValueError):
        compute_metrics(y_true, pred_b_wrong_length)
    # The resampling-based functions index into the mismatched array with
    # positions valid for `y_true`'s length, which can land out of bounds
    # for the shorter `pred_b_wrong_length` array (IndexError) as well as
    # hitting a plain shape-mismatch (ValueError) depending on which index
    # is drawn -- either is an acceptably loud failure; silently succeeding
    # would not be.
    with pytest.raises((ValueError, IndexError)):
        paired_bootstrap_ci(y_true, pred_a, pred_b_wrong_length, rmse)
    with pytest.raises((ValueError, IndexError)):
        paired_permutation_test(y_true, pred_a, pred_b_wrong_length, rmse)


def test_ab_metrics_accept_paired_equal_length_predictions():
    rng = np.random.RandomState(0)
    y_true = rng.uniform(100_000, 900_000, size=50)
    pred_a = y_true + rng.normal(0, 50_000, size=50)
    pred_b = y_true + rng.normal(0, 20_000, size=50)
    ci = paired_bootstrap_ci(y_true, pred_a, pred_b, rmse, n_boot=200)
    assert ci.n_bootstrap == 200
    assert len(ci.bootstrap_ci_95) == 2
    perm = paired_permutation_test(y_true, pred_a, pred_b, rmse, n_perm=200)
    assert 0 < perm.p_value <= 1.0
