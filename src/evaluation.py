"""Evaluation protocol: splitting, metrics, and the statistical A/B test.

Protocol (defined BEFORE looking at any final-test result, per the
requirement to avoid test-set peeking):

  1. Group-aware split by house `id` (see features.py finding #2):
     70% of houses -> development set, 30% -> locked final test set.
  2. All hyperparameter selection and ablation-stage comparison happens via
     5-fold GroupKFold cross-validation INSIDE the development set only.
  3. The locked test set is touched exactly once per model, at the very end,
     to produce the numbers in the final A/B table.
  4. Because A and B predict the SAME locked test houses, the comparison is
     paired. We separate three distinct statistical questions rather than
     conflating them (see post-review correction below):

       (a) EFFECT-SIZE UNCERTAINTY per metric (RMSE, MAE, R2) -> paired
           bootstrap 95% confidence interval, computed independently for
           each metric.
       (b) FORMAL HYPOTHESIS TEST of H0 ("A and B are exchangeable", i.e.
           no systematic difference) per metric -> a paired permutation
           (randomization) test, which is the textbook-correct test for
           "would relabeling which model produced which prediction, on
           each paired observation, plausibly produce a metric gap this
           large by chance". This does NOT assume normality and directly
           tests the same H0 the bootstrap CI is silent about (a CI
           excluding zero is suggestive of significance but is not itself
           a hypothesis test).
       (c) A SEPARATE, secondary, distribution-free check on paired
           per-house absolute errors (not on RMSE/MAE/R2 directly) using
           the Wilcoxon signed-rank test. This tests a different but
           related null ("the two models' absolute errors on the same
           house come from the same distribution") and is reported once,
           not attached identically to every metric.

============================================================
POST-REVIEW CORRECTIONS
============================================================
1. LEAKAGE: bedroom-cap/ratio-median fitting moved into a proper
   `sklearn` transformer fit per-CV-fold (see features.py).
2. BOOTSTRAP P-VALUE: a percentile bootstrap with 0 exceedances out of
   n_boot resamples does NOT mean the true p-value is exactly 0 -- it means
   only that p < 1/(n_boot+1). We now report the finite-sample-corrected
   Monte Carlo p-value, p = (extreme_count + 1) / (n_boot + 1), which is
   never exactly zero and is upper-bounded honestly.
3. WILCOXON SEPARATION: the Wilcoxon signed-rank result (always computed on
   paired |error|, regardless of which headline metric is being reported)
   is no longer duplicated identically under the RMSE/MAE/R2 entries. It is
   computed once and reported under its own key.
4. PAIRED PERMUTATION TEST added as the primary hypothesis test per metric,
   with the bootstrap CI now described purely as an effect-size interval.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from scipy import stats

RANDOM_SEED = 42
N_SPLITS = 5
N_BOOTSTRAP = 5000
N_PERMUTATIONS = 5000


def group_train_test_split(
    df: pd.DataFrame, group_col: str = "id", test_frac: float = 0.30, seed: int = RANDOM_SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.RandomState(seed)
    unique_groups = df[group_col].unique()
    rng.shuffle(unique_groups)
    n_test_groups = int(len(unique_groups) * test_frac)
    test_groups = set(unique_groups[:n_test_groups])
    is_test = df[group_col].isin(test_groups)
    return df.loc[~is_test].reset_index(drop=True), df.loc[is_test].reset_index(drop=True)


def make_group_kfold(df: pd.DataFrame, group_col: str = "id", n_splits: int = N_SPLITS):
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(df, groups=df[group_col]))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "mape_pct": mape(y_true, y_pred),
    }


def monte_carlo_p_value(extreme_count: int, n_resamples: int) -> float:
    """Finite-sample-corrected Monte Carlo p-value. Never returns exactly 0
    (the smallest value it can return is 1/(n_resamples+1)), which is the
    honest statement of resolution for a resampling test: 0 exceedances out
    of N resamples means p < 1/(N+1), not p = 0."""
    return float((extreme_count + 1) / (n_resamples + 1))


@dataclass
class BootstrapCI:
    metric_a: float
    metric_b: float
    abs_delta: float
    relative_delta_pct: float
    bootstrap_ci_95: Tuple[float, float]
    n_bootstrap: int


@dataclass
class PermutationTestResult:
    observed_delta: float
    p_value: float
    n_permutations: int


@dataclass
class WilcoxonResult:
    statistic: float
    p_value: float
    p_value_underflowed: bool
    p_value_report: str
    n_paired_obs: int


def paired_bootstrap_ci(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
    lower_is_better: bool = True,
) -> BootstrapCI:
    """Paired bootstrap over the SAME resampled houses for both models.

    This function answers ONLY the effect-size-uncertainty question ("how
    wide is our uncertainty about the size of the improvement"). The formal
    H0 test is `paired_permutation_test` below, not this CI -- see the
    module docstring's "POST-REVIEW CORRECTIONS" #4.
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        m_a = metric_fn(y_true[idx], pred_a[idx])
        m_b = metric_fn(y_true[idx], pred_b[idx])
        deltas[i] = (m_a - m_b) if lower_is_better else (m_b - m_a)
    ci = (float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)))

    metric_a = metric_fn(y_true, pred_a)
    metric_b = metric_fn(y_true, pred_b)
    abs_delta = (metric_a - metric_b) if lower_is_better else (metric_b - metric_a)
    relative_delta_pct = 100 * abs_delta / metric_a if metric_a != 0 else float("nan")

    return BootstrapCI(
        metric_a=metric_a, metric_b=metric_b, abs_delta=abs_delta,
        relative_delta_pct=relative_delta_pct, bootstrap_ci_95=ci, n_bootstrap=n_boot,
    )


def paired_permutation_test(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_perm: int = N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
    lower_is_better: bool = True,
) -> PermutationTestResult:
    """Paired permutation (randomization) test for H0: A and B are
    exchangeable on each house (i.e. which model produced which prediction
    for a given house carries no information). For each of n_perm
    iterations, independently for each house, the A/B prediction labels are
    swapped with probability 0.5, the metric is recomputed for the
    resulting two pseudo-groups, and the same delta statistic used for the
    real data is recorded. The two-sided p-value is the fraction of
    permuted deltas at least as extreme (by absolute value) as the observed
    one, with the standard +1/+1 finite-sample correction (see
    `monte_carlo_p_value`).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)

    metric_a = metric_fn(y_true, pred_a)
    metric_b = metric_fn(y_true, pred_b)
    observed = (metric_a - metric_b) if lower_is_better else (metric_b - metric_a)

    extreme_count = 0
    for _ in range(n_perm):
        swap = rng.rand(n) < 0.5
        perm_a = np.where(swap, pred_b, pred_a)
        perm_b = np.where(swap, pred_a, pred_b)
        m_a = metric_fn(y_true, perm_a)
        m_b = metric_fn(y_true, perm_b)
        delta = (m_a - m_b) if lower_is_better else (m_b - m_a)
        if abs(delta) >= abs(observed):
            extreme_count += 1

    p_value = monte_carlo_p_value(extreme_count, n_perm)
    return PermutationTestResult(observed_delta=observed, p_value=p_value, n_permutations=n_perm)


def wilcoxon_on_paired_absolute_errors(
    y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray
) -> WilcoxonResult:
    """Secondary, distribution-free test on paired |error| (NOT on
    RMSE/MAE/R2 directly -- those are tested with `paired_permutation_test`
    above). Valid here because A and B are evaluated on the identical set
    of houses, and this test makes no normality assumption, which matters
    given the heavy right skew documented in the data audit.

    POST-REVIEW FIX: scipy's default asymptotic-normal p-value calculation
    can underflow to an exact double-precision 0.0 at large n / large test
    statistics (as happens here). That underflow is a floating-point
    representation limit, not evidence the true p-value is exactly zero, so
    we never report a bare 0.0. When scipy returns 0.0, we report
    `p_value_report = "< machine precision"` -- deliberately NOT a specific
    numeric bound like "< 1e-16", since that specific threshold is not
    something this function actually verifies; "below machine precision" is
    the claim the underflow itself supports."""
    err_a = np.abs(y_true - pred_a)
    err_b = np.abs(y_true - pred_b)
    wstat, wp = stats.wilcoxon(err_a, err_b)
    wp = float(wp)
    underflowed = (wp == 0.0)
    report = "< machine precision (scipy reported an exact 0.0; true p-value is nonzero but unrepresentable at double precision)" if underflowed else f"{wp:.6g}"
    return WilcoxonResult(
        statistic=float(wstat), p_value=wp, p_value_underflowed=underflowed,
        p_value_report=report, n_paired_obs=len(y_true),
    )


def full_ab_report(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> Dict:
    """Builds the complete, cleanly-separated A/B statistical report:
    per-metric bootstrap CI + permutation-test p-value for RMSE/MAE/R2,
    plus ONE shared Wilcoxon result on paired absolute errors."""
    metric_specs = [("rmse", rmse, True), ("mae", mae, True), ("r2", r2, False)]
    metric_comparisons = {}
    for name, fn, lower_is_better in metric_specs:
        ci = paired_bootstrap_ci(y_true, pred_a, pred_b, fn, lower_is_better=lower_is_better)
        perm = paired_permutation_test(y_true, pred_a, pred_b, fn, lower_is_better=lower_is_better)
        metric_comparisons[name] = {
            "metric_a": ci.metric_a,
            "metric_b": ci.metric_b,
            "abs_delta": ci.abs_delta,
            "relative_delta_pct": ci.relative_delta_pct,
            "bootstrap_ci_95": ci.bootstrap_ci_95,
            "n_bootstrap": ci.n_bootstrap,
            "permutation_test": {
                "observed_delta": perm.observed_delta,
                "p_value": perm.p_value,
                "n_permutations": perm.n_permutations,
            },
        }

    wilcoxon = wilcoxon_on_paired_absolute_errors(y_true, pred_a, pred_b)
    return {
        "metric_comparisons": metric_comparisons,
        "paired_absolute_error_test": {
            "test": "Wilcoxon signed-rank",
            "statistic": wilcoxon.statistic,
            "p_value_raw": wilcoxon.p_value,
            "p_value_underflowed": wilcoxon.p_value_underflowed,
            "p_value_report": wilcoxon.p_value_report,
            "n_paired_obs": wilcoxon.n_paired_obs,
        },
    }
