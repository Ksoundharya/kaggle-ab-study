"""Post-hoc diagnostics on the locked test set: residual analysis, error
slicing by subgroup, permutation importance, and seed/fold sensitivity.
Reads the artifacts written by run_experiment.py; does not refit A or the
final B model except where a robustness check explicitly requires a refit
under a different seed.

POST-REVIEW CHANGE: the seed-robustness check now re-splits raw data (not
pre-engineered data) per seed, fits BOTH A and B fresh on each seed's own
development set, and reports each seed's LOCKED-TEST-SET delta between A
and B -- not just B's CV RMSE in isolation. This directly answers "is B's
advantage over A stable across seeds", which is what the assignment's
robustness section is actually asking, rather than only "is B's absolute
RMSE stable".
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_raw, TARGET
from features import (
    build_baseline_preprocessor, build_engineered_preprocessor,
    RAW_COLUMNS_FOR_BASELINE, RAW_COLUMNS_FOR_ENGINEERED,
)
from evaluation import group_train_test_split, make_group_kfold, compute_metrics, rmse, RANDOM_SEED, N_SPLITS
from models import baseline_model, b3_gradient_boosting

OUT = Path(__file__).resolve().parents[1] / "outputs"


def log(msg):
    print(f"[diagnostics] {msg}", flush=True)


def main():
    preds = pd.read_csv(OUT / "test_predictions.csv")
    y = preds[TARGET].values
    pred_a = preds["pred_A"].values
    pred_b = preds["pred_B"].values

    # ---------- Residual diagnostics ----------
    resid_a = y - pred_a
    resid_b = y - pred_b
    diag = {
        "A": {
            "residual_mean": float(resid_a.mean()),
            "residual_std": float(resid_a.std()),
            "residual_skew": float(pd.Series(resid_a).skew()),
            "pct_within_10pct_of_true": float((np.abs(resid_a) / y < 0.10).mean() * 100),
            "worst_5_abs_errors": sorted(np.abs(resid_a))[-5:],
        },
        "B": {
            "residual_mean": float(resid_b.mean()),
            "residual_std": float(resid_b.std()),
            "residual_skew": float(pd.Series(resid_b).skew()),
            "pct_within_10pct_of_true": float((np.abs(resid_b) / y < 0.10).mean() * 100),
            "worst_5_abs_errors": sorted(np.abs(resid_b))[-5:],
        },
    }
    diag["A"]["corr_abs_resid_vs_pred"] = float(np.corrcoef(np.abs(resid_a), pred_a)[0, 1])
    diag["B"]["corr_abs_resid_vs_pred"] = float(np.corrcoef(np.abs(resid_b), pred_b)[0, 1])

    # ---------- Error slicing ----------
    def slice_metrics(mask, name):
        if mask.sum() < 20:
            return None
        return {
            "segment": name,
            "n": int(mask.sum()),
            "rmse_A": rmse(y[mask], pred_a[mask]),
            "rmse_B": rmse(y[mask], pred_b[mask]),
            "delta_rmse_A_minus_B": float(rmse(y[mask], pred_a[mask]) - rmse(y[mask], pred_b[mask])),
        }

    slices = []
    grade_bins = [(1, 6, "grade<=6 (below average)"), (7, 7, "grade=7 (average)"),
                  (8, 9, "grade 8-9 (good)"), (10, 13, "grade>=10 (excellent/luxury)")]
    for lo, hi, name in grade_bins:
        mask = (preds["grade"] >= lo) & (preds["grade"] <= hi)
        r = slice_metrics(mask.values, name)
        if r:
            slices.append(r)

    waterfront_bins = [(0, "no waterfront"), (1, "waterfront")]
    for val, name in waterfront_bins:
        mask = preds["waterfront"] == val
        r = slice_metrics(mask.values, name)
        if r:
            slices.append(r)

    age_bins = [(0, 20, "built <20y before sale"), (20, 50, "built 20-50y before sale"),
                (50, 200, "built 50y+ before sale")]
    house_age = 2015 - preds["yr_built"]  # dataset spans May 2014-May 2015; 2015 is a fixed reference,
    # consistent across all rows, adequate for a descriptive age bucket (not used in any model).
    for lo, hi, name in age_bins:
        mask = (house_age >= lo) & (house_age < hi)
        r = slice_metrics(mask.values, name)
        if r:
            slices.append(r)

    price_bins = [(0, 300000, "price < $300k"), (300000, 600000, "$300k-$600k"),
                  (600000, 1000000, "$600k-$1M"), (1000000, 1e9, "price > $1M")]
    for lo, hi, name in price_bins:
        mask = (preds[TARGET] >= lo) & (preds[TARGET] < hi)
        r = slice_metrics(mask.values, name)
        if r:
            slices.append(r)

    condition_bins = [(1, 2, "condition 1-2 (poor)"), (3, 3, "condition 3 (average)"),
                       (4, 5, "condition 4-5 (good/very good)")]
    for lo, hi, name in condition_bins:
        mask = (preds["condition"] >= lo) & (preds["condition"] <= hi)
        r = slice_metrics(mask.values, name)
        if r:
            slices.append(r)

    with open(OUT / "diagnostics.json", "w") as f:
        json.dump(diag, f, indent=2, default=str)
    with open(OUT / "error_slices.json", "w") as f:
        json.dump(slices, f, indent=2, default=str)
    log(f"Wrote diagnostics.json and error_slices.json ({len(slices)} segments)")

    # ---------- Permutation importance for B on the locked test set ----------
    log("Refitting B on dev set for permutation importance")
    raw = load_raw()
    dev_df, test_df = group_train_test_split(raw, group_col="id", test_frac=0.30, seed=RANDOM_SEED)
    feature_cols = RAW_COLUMNS_FOR_ENGINEERED
    best_params = json.load(open(OUT / "ablation_cv.json"))["B3_best_params"]

    pipe = Pipeline(
        build_engineered_preprocessor(for_linear=False).steps
        + [("model", b3_gradient_boosting(**best_params))]
    )
    pipe.fit(dev_df[feature_cols], np.log1p(dev_df[TARGET]))

    t0 = time.perf_counter()
    rng = np.random.RandomState(RANDOM_SEED)
    sample_idx = rng.choice(len(test_df), size=min(1500, len(test_df)), replace=False)
    X_sample = test_df.iloc[sample_idx][feature_cols]
    y_sample = np.log1p(test_df.iloc[sample_idx][TARGET])
    perm = permutation_importance(
        pipe, X_sample, y_sample, n_repeats=5, random_state=RANDOM_SEED,
        scoring="neg_root_mean_squared_error", n_jobs=-1,
    )
    perm_time = time.perf_counter() - t0

    ohe = pipe.named_steps["select"].named_transformers_["cat"].named_steps["onehot"]
    zip_names = [f"zipcode_{c}" for c in ohe.categories_[0]]
    all_names = feature_cols[:-1] + zip_names  # raw numeric cols + expanded zipcode names
    importances = list(zip(all_names, perm.importances_mean))
    zip_importance = sum(v for n, v in importances if n.startswith("zipcode_"))
    non_zip = [(n, v) for n, v in importances if not n.startswith("zipcode_")]
    non_zip.append(("zipcode (aggregated over 70 dummies)", zip_importance))
    non_zip.sort(key=lambda x: -x[1])

    importance_out = {
        "method": "permutation_importance (neg_root_mean_squared_error drop), n_repeats=5, "
                  "1500-row test subsample, seed=42",
        "compute_time_sec": perm_time,
        "ranking": [{"feature": n, "importance_mean_rmse_increase": float(v)} for n, v in non_zip],
    }
    with open(OUT / "importance.json", "w") as f:
        json.dump(importance_out, f, indent=2)
    log(f"Top 5 features by permutation importance: {[n for n,_ in non_zip[:5]]}")

    # ---------- Robustness: A vs B under 5 different dev/test splits ----------
    log("Robustness: re-splitting raw data under 5 seeds, fitting BOTH A and B fresh, "
        "comparing on each seed's own locked test set")
    seed_results = []
    for seed in [0, 1, 7, 42, 123]:
        dev_s, test_s = group_train_test_split(raw, group_col="id", test_frac=0.30, seed=seed)

        pipe_a_s = Pipeline(build_baseline_preprocessor().steps + [("model", baseline_model())])
        pipe_a_s.fit(dev_s[RAW_COLUMNS_FOR_BASELINE], np.log1p(dev_s[TARGET]))
        pred_a_s = np.expm1(pipe_a_s.predict(test_s[RAW_COLUMNS_FOR_BASELINE]))

        pipe_b_s = Pipeline(
            build_engineered_preprocessor(for_linear=False).steps
            + [("model", b3_gradient_boosting(**best_params))]
        )
        pipe_b_s.fit(dev_s[feature_cols], np.log1p(dev_s[TARGET]))
        pred_b_s = np.expm1(pipe_b_s.predict(test_s[feature_cols]))

        rmse_a_s = rmse(test_s[TARGET].values, pred_a_s)
        rmse_b_s = rmse(test_s[TARGET].values, pred_b_s)
        delta = rmse_a_s - rmse_b_s
        pct_improvement = 100 * delta / rmse_a_s

        seed_results.append({
            "seed": seed,
            "rmse_A": rmse_a_s,
            "rmse_B": rmse_b_s,
            "delta_rmse_A_minus_B": float(delta),
            "pct_improvement": float(pct_improvement),
        })
        log(f"  seed={seed}  A={rmse_a_s:.0f}  B={rmse_b_s:.0f}  "
            f"delta={delta:.0f}  improvement={pct_improvement:.1f}%")

    with open(OUT / "robustness_seeds.json", "w") as f:
        json.dump(seed_results, f, indent=2)


if __name__ == "__main__":
    main()
