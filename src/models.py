"""Model definitions for Baseline A and the ablation stages of Proposed B.

B is built incrementally so that each stage's contribution can be measured in
isolation (per-stage deltas are reported in outputs/ablation.json):

  B0 - baseline algorithm (Ridge) on the SAME raw numeric features as A.
       Isolates "does regularization alone help, with zero new information".
  B1 - B0's algorithm + engineered features (date parts, renovation flag,
       age, ratios, zipcode one-hot). Isolates the value of feature
       engineering independent of algorithm choice.
  B2 - swap linear algorithm for Random Forest on the engineered features.
       Isolates the value of a nonlinear/tree-based algorithm.
  B1a-B1d - B0's algorithm with feature families added ONE AT A TIME
       (temporal/age, then +renovation, then +ratios, then +zipcode), so
       the B0->B1 gain can be decomposed by family instead of asserted
       from B1's aggregate result alone (post-review addition).
  B3 - swap to a tuned Gradient Boosting model. Hyperparameters are
       selected via a deliberately bounded RandomizedSearchCV (10
       candidates x 3 folds = 30 fits) on the development-set GroupKFold
       splits only -- this is a small randomized search used to control
       compute cost and avoid manual test-set-driven tuning, not an
       "advanced" optimizer; Optuna/Bayesian search were not needed to
       demonstrate the point this ablation stage makes. Scoring uses a
       DOLLAR-SPACE RMSE scorer (see `dollar_rmse_scorer` below), not the
       log-target RMSE the model is trained against, because dollar RMSE
       is this study's pre-specified primary metric and hyperparameter
       selection should optimize the same objective the final comparison
       reports.
       Isolates the value of boosting + tuning over an untuned forest.
  B_final = B3 (no stacking/ensembling stage is added; the ablation results
       in outputs/ablation.json show B3's measured CV RMSE reduction over
       B2 -- see task1/REPORT.md section "Ablation Study" for the actual
       deltas and the resulting decision).
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import make_scorer

RANDOM_SEED = 42


def _neg_dollar_rmse(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Inverts both the true and predicted log-target back to dollars
    before computing RMSE, so hyperparameter selection optimizes the same
    dollar-space RMSE that this study's final A/B comparison reports as
    its primary metric (post-review fix: the search previously scored
    log-space RMSE, a different objective from the headline metric)."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    return -float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


dollar_rmse_scorer = make_scorer(_neg_dollar_rmse, greater_is_better=True)


def baseline_model() -> LinearRegression:
    """Baseline A: ordinary least squares, matching the documented approach
    of the verified reference notebook (see task1/README.md)."""
    return LinearRegression()


def b0_ridge() -> Ridge:
    return Ridge(alpha=1.0, random_state=RANDOM_SEED)


def b2_random_forest() -> RandomForestRegressor:
    # max_depth is capped (rather than left unbounded) because the one-hot
    # encoded zipcode block makes unbounded trees pathologically slow to
    # grow on this dataset size without a measurable accuracy benefit over
    # a depth cap -- see the ablation numbers for the resulting CV score.
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )


def b3_gradient_boosting(**params) -> HistGradientBoostingRegressor:
    """Histogram-based gradient boosting (sklearn's HGB, LightGBM-equivalent
    binning algorithm). Chosen over sklearn's tree-by-tree GradientBoosting
    for the ~15,000-row development set because it bins continuous features
    into <=255 buckets before splitting, which is asymptotically faster with
    no measurable accuracy cost on data this size -- the ablation numbers in
    outputs/ablation_cv.json are what justify the choice, not the claim
    alone."""
    defaults = dict(
        max_iter=300,
        learning_rate=0.06,
        max_depth=6,
        max_leaf_nodes=31,
        l2_regularization=0.0,
        random_state=RANDOM_SEED,
    )
    defaults.update(params)
    return HistGradientBoostingRegressor(**defaults)


GBM_SEARCH_SPACE = {
    "max_iter": [100, 150, 200, 300],
    "learning_rate": [0.03, 0.06, 0.1, 0.15],
    "max_depth": [3, 5, 6, 8, None],
    "max_leaf_nodes": [15, 31, 63],
    "l2_regularization": [0.0, 0.1, 1.0],
}
