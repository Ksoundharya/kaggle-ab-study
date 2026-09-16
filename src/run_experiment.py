"""End-to-end Task 1 experiment: audit -> split -> baseline A -> ablation ->
proposed B -> statistical A/B test -> diagnostics -> error slicing ->
importance -> robustness -> cost. Every number below is computed from a real
execution against the real dataset in task1/data/kc_house_data.csv; nothing
here is hand-typed as a result.

POST-REVIEW CHANGE: the raw/test split now happens BEFORE any feature
engineering or bedroom-cap fitting (previously the cap was computed from the
full dataset, which is leakage -- see features.py's "LEAKAGE FIX" docstring).
Every pipeline below starts with `HouseFeatureEngineer`, so its `.fit()` is
called fresh on each CV training fold and on the development set alone for
the final model, never on the locked test set.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import scipy
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GroupKFold

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_raw, audit, save_audit, TARGET
from features import (
    build_baseline_preprocessor, build_engineered_preprocessor,
    RAW_COLUMNS_FOR_BASELINE, RAW_COLUMNS_FOR_ENGINEERED, feature_columns_for,
)
from evaluation import (
    group_train_test_split, make_group_kfold, compute_metrics, rmse, mae, r2,
    full_ab_report, RANDOM_SEED, N_SPLITS,
)
from models import (
    baseline_model, b0_ridge, b2_random_forest, b3_gradient_boosting,
    GBM_SEARCH_SPACE, dollar_rmse_scorer,
)

OUT = Path(__file__).resolve().parents[1] / "outputs"
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[run_experiment] {msg}", flush=True)


def cv_evaluate(pipeline_builder, df, feature_cols, folds, use_log_target=True):
    """Cross-validate a fresh pipeline instance per fold on the dev set.
    Because `pipeline_builder()` returns a pipeline whose first step is
    `HouseFeatureEngineer`, calling `.fit(train[feature_cols], ...)` fits
    the bedroom-cap/ratio-median constants on THIS FOLD'S training rows
    only; `.predict(val[feature_cols])` applies those already-fit
    constants to the validation fold without recomputing anything from it.
    """
    fold_metrics = []
    for i, (tr_idx, va_idx) in enumerate(folds):
        t_fold = time.perf_counter()
        train, val = df.iloc[tr_idx], df.iloc[va_idx]
        pipe = pipeline_builder()
        y_tr = np.log1p(train[TARGET]) if use_log_target else train[TARGET]
        pipe.fit(train[feature_cols], y_tr)
        pred = pipe.predict(val[feature_cols])
        pred = np.expm1(pred) if use_log_target else pred
        fold_metrics.append(compute_metrics(val[TARGET].values, pred))
        log(f"  fold {i+1}/{len(folds)} done in {time.perf_counter()-t_fold:.1f}s "
            f"rmse={fold_metrics[-1]['rmse']:.0f}")
    keys = fold_metrics[0].keys()
    agg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    agg_std = {f"{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in keys}
    return {**agg, **agg_std, "fold_metrics": fold_metrics}


def main():
    t0 = time.perf_counter()
    raw = load_raw()
    audit_result = audit(raw)
    save_audit(audit_result, OUT / "audit.json")
    log(f"Audit complete. rows={audit_result.n_rows} dup_ids={audit_result.n_duplicate_ids}")

    # ---- Split BEFORE any feature engineering (post-review leakage fix) ----
    dev_df, test_df = group_train_test_split(raw, group_col="id", test_frac=0.30, seed=RANDOM_SEED)
    log(f"Dev set: {len(dev_df)} rows / {dev_df['id'].nunique()} houses; "
        f"Test set: {len(test_df)} rows / {test_df['id'].nunique()} houses")
    overlap = set(dev_df["id"]) & set(test_df["id"])
    assert len(overlap) == 0, "Group split leaked house ids across dev/test"

    folds = make_group_kfold(dev_df, group_col="id", n_splits=N_SPLITS)

    baseline_cols = RAW_COLUMNS_FOR_BASELINE
    engineered_cols = RAW_COLUMNS_FOR_ENGINEERED

    results = {}

    # ---- Baseline A ----
    log("CV: Baseline A (OLS, raw numeric features, reference-aligned)")
    results["A_baseline_ols"] = cv_evaluate(
        lambda: Pipeline(build_baseline_preprocessor().steps + [("model", baseline_model())]),
        dev_df, baseline_cols, folds,
    )

    # ---- Ablation B0: Ridge, same raw features as A ----
    log("CV: B0 (Ridge, raw numeric features)")
    results["B0_ridge_raw_features"] = cv_evaluate(
        lambda: Pipeline(build_baseline_preprocessor().steps + [("model", b0_ridge())]),
        dev_df, baseline_cols, folds,
    )

    # ---- Feature-family ablation: B1a -> B1d (post-review addition) ----
    family_stages = [
        ("B1a_ridge_plus_temporal_age", dict(include_temporal_age=True, include_renovation=False,
                                              include_ratios=False, include_zipcode=False)),
        ("B1b_ridge_plus_renovation", dict(include_temporal_age=True, include_renovation=True,
                                            include_ratios=False, include_zipcode=False)),
        ("B1c_ridge_plus_ratios", dict(include_temporal_age=True, include_renovation=True,
                                        include_ratios=True, include_zipcode=False)),
        ("B1d_ridge_plus_zipcode", dict(include_temporal_age=True, include_renovation=True,
                                         include_ratios=True, include_zipcode=True)),
    ]
    for stage_name, flags in family_stages:
        log(f"CV: {stage_name}")
        cols = feature_columns_for(**flags)
        results[stage_name] = cv_evaluate(
            lambda flags=flags: Pipeline(
                build_engineered_preprocessor(for_linear=True, **flags).steps + [("model", b0_ridge())]
            ),
            dev_df, cols, folds,
        )
    # Keep the name "B1" as an alias for the full-feature-set linear stage
    # (B1d), for continuity with the original ablation narrative.
    results["B1_ridge_engineered_features"] = results["B1d_ridge_plus_zipcode"]

    # ---- Ablation B2: Random Forest, engineered features ----
    log("CV: B2 (Random Forest, engineered features)")
    results["B2_random_forest_engineered"] = cv_evaluate(
        lambda: Pipeline(build_engineered_preprocessor(for_linear=False).steps + [("model", b2_random_forest())]),
        dev_df, engineered_cols, folds,
    )

    # ---- Ablation B3: Gradient Boosting, tuned, engineered features ----
    log("Hyperparameter search: B3 (bounded RandomizedSearchCV, dollar-space RMSE scorer)")
    search_pipe = Pipeline(
        build_engineered_preprocessor(for_linear=False).steps + [("model", b3_gradient_boosting())]
    )
    param_dist = {f"model__{k}": v for k, v in GBM_SEARCH_SPACE.items()}
    gkf = GroupKFold(n_splits=3)  # 3 folds for the search only, to bound wall-clock cost;
    # the winning config is then re-evaluated on the full 5-fold protocol below,
    # so the reported ablation numbers still use the same protocol as every other stage.
    search = RandomizedSearchCV(
        search_pipe, param_distributions=param_dist, n_iter=10,
        scoring=dollar_rmse_scorer, cv=gkf, n_jobs=-1,
        random_state=RANDOM_SEED, refit=False, verbose=1,
    )
    y_dev_log = np.log1p(dev_df[TARGET])
    search.fit(dev_df[engineered_cols], y_dev_log, groups=dev_df["id"])
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    log(f"Best GBM params (selected on dollar-space RMSE): {best_params}")

    results["B3_gradient_boosting_tuned"] = cv_evaluate(
        lambda: Pipeline(
            build_engineered_preprocessor(for_linear=False).steps
            + [("model", b3_gradient_boosting(**best_params))]
        ),
        dev_df, engineered_cols, folds,
    )
    results["B3_best_params"] = best_params

    with open(OUT / "ablation_cv.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ---- Final locked test-set evaluation: A vs B_final (B3) ----
    log("Final locked-test evaluation: A vs B_final")
    pipe_a = Pipeline(build_baseline_preprocessor().steps + [("model", baseline_model())])
    pipe_a.fit(dev_df[baseline_cols], np.log1p(dev_df[TARGET]))
    pred_a_test = np.expm1(pipe_a.predict(test_df[baseline_cols]))

    t_train_b0 = time.perf_counter()
    pipe_b = Pipeline(
        build_engineered_preprocessor(for_linear=False).steps
        + [("model", b3_gradient_boosting(**best_params))]
    )
    pipe_b.fit(dev_df[engineered_cols], np.log1p(dev_df[TARGET]))
    train_time_b = time.perf_counter() - t_train_b0

    t_inf_b0 = time.perf_counter()
    pred_b_test = np.expm1(pipe_b.predict(test_df[engineered_cols]))
    infer_time_b = time.perf_counter() - t_inf_b0

    t_train_a0 = time.perf_counter()
    Pipeline(build_baseline_preprocessor().steps + [("model", baseline_model())]).fit(
        dev_df[baseline_cols], np.log1p(dev_df[TARGET])
    )
    train_time_a = time.perf_counter() - t_train_a0

    y_test = test_df[TARGET].values

    final_metrics_a = compute_metrics(y_test, pred_a_test)
    final_metrics_b = compute_metrics(y_test, pred_b_test)

    ab_report = full_ab_report(y_test, pred_a_test, pred_b_test)

    final = {
        "final_metrics_A": final_metrics_a,
        "final_metrics_B": final_metrics_b,
        "ab_test": ab_report,
        "train_time_sec_A": train_time_a,
        "train_time_sec_B": train_time_b,
        "inference_time_sec_B_full_test_set": infer_time_b,
        "n_test_rows": len(test_df),
        "n_dev_rows": len(dev_df),
    }
    with open(OUT / "final_ab_results.json", "w") as f:
        json.dump(final, f, indent=2, default=str)

    # Persist predictions for downstream diagnostics/error-slicing/robustness scripts
    test_out = test_df[["id", TARGET, "zipcode", "grade", "condition", "sqft_living",
                         "waterfront", "yr_built"]].copy()
    test_out["pred_A"] = pred_a_test
    test_out["pred_B"] = pred_b_test
    test_out.to_csv(OUT / "test_predictions.csv", index=False)

    dev_df.to_csv(OUT / "dev_df_cache.csv", index=False)

    # ---- Environment / dependency record (post-review reproducibility fix) ----
    env_record = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "scipy_version": scipy.__version__,
        "random_seed": RANDOM_SEED,
    }
    with open(OUT / "environment.json", "w") as f:
        json.dump(env_record, f, indent=2)

    total_time = time.perf_counter() - t0
    log(f"Done in {total_time:.1f}s. Final A RMSE={final_metrics_a['rmse']:.0f} "
        f"B RMSE={final_metrics_b['rmse']:.0f}  R2 A={final_metrics_a['r2']:.4f} B={final_metrics_b['r2']:.4f}")
    log(f"Environment: {env_record}")


if __name__ == "__main__":
    main()
