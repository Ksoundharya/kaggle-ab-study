"""Generates every figure referenced in task1/REPORT.md. Reads only the JSON/
CSV artifacts already written by run_experiment.py and diagnostics.py (plus
the raw CSV for the two EDA plots) -- no numbers are computed fresh here that
aren't already substantiated elsewhere; this script is presentation-only.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_raw, TARGET
from evaluation import paired_bootstrap_ci, rmse

OUT = Path(__file__).resolve().parents[1] / "outputs"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.grid": True, "grid.alpha": 0.3, "font.size": 10,
})
COLOR_A = "#d9534f"   # baseline A
COLOR_B = "#2a6f97"   # proposed B


def savefig(name):
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"[make_plots] wrote {path}")


def fig_eda_price_distribution():
    raw = load_raw()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(raw[TARGET], bins=80, color=COLOR_B, alpha=0.85)
    axes[0].set_title("Sale price distribution (raw)")
    axes[0].set_xlabel("price ($)")
    axes[0].set_ylabel("count")
    axes[1].hist(np.log1p(raw[TARGET]), bins=80, color=COLOR_B, alpha=0.85)
    axes[1].set_title("log1p(price) distribution")
    axes[1].set_xlabel("log1p(price)")
    fig.suptitle("EDA: target distribution before/after log transform (skew=4.02 raw)")
    savefig("01_eda_price_distribution.png")


def fig_eda_price_vs_sqft_and_grade():
    raw = load_raw()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(raw["sqft_living"], raw[TARGET], s=4, alpha=0.15, color=COLOR_B)
    axes[0].set_xlabel("sqft_living")
    axes[0].set_ylabel("price ($)")
    axes[0].set_title("Price vs. living area")
    grade_means = raw.groupby("grade")[TARGET].mean()
    axes[1].bar(grade_means.index.astype(str), grade_means.values, color=COLOR_B)
    axes[1].set_xlabel("grade")
    axes[1].set_ylabel("mean price ($)")
    axes[1].set_title("Mean price by King County grade")
    fig.suptitle("EDA: two of the strongest bivariate relationships with price")
    savefig("02_eda_price_relationships.png")


def fig_actual_vs_predicted():
    preds = pd.read_csv(OUT / "test_predictions.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    lims = (0, preds[TARGET].quantile(0.995))
    for ax, col, label, color in [
        (axes[0], "pred_A", "Baseline A (OLS)", COLOR_A),
        (axes[1], "pred_B", "Proposed B (tuned HGB)", COLOR_B),
    ]:
        ax.scatter(preds[TARGET], preds[col], s=4, alpha=0.2, color=color)
        ax.plot(lims, lims, "k--", linewidth=1, label="perfect prediction")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("actual price ($)")
        ax.set_ylabel("predicted price ($)")
        ax.set_title(label)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle("Actual vs. predicted price on the locked test set (6,485 houses)")
    savefig("03_actual_vs_predicted.png")


def fig_residuals():
    preds = pd.read_csv(OUT / "test_predictions.csv")
    resid_a = preds[TARGET] - preds["pred_A"]
    resid_b = preds[TARGET] - preds["pred_B"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].hist(resid_a, bins=80, color=COLOR_A, alpha=0.85)
    axes[0, 0].set_title("Baseline A residual distribution")
    axes[0, 0].axvline(0, color="k", linewidth=1)
    axes[0, 1].hist(resid_b, bins=80, color=COLOR_B, alpha=0.85)
    axes[0, 1].set_title("Proposed B residual distribution")
    axes[0, 1].axvline(0, color="k", linewidth=1)
    axes[1, 0].scatter(preds["pred_A"], resid_a, s=4, alpha=0.15, color=COLOR_A)
    axes[1, 0].axhline(0, color="k", linewidth=1)
    axes[1, 0].set_xlabel("predicted price ($)")
    axes[1, 0].set_ylabel("residual ($)")
    axes[1, 0].set_title("A: residuals vs. prediction (heteroscedasticity check)")
    axes[1, 1].scatter(preds["pred_B"], resid_b, s=4, alpha=0.15, color=COLOR_B)
    axes[1, 1].axhline(0, color="k", linewidth=1)
    axes[1, 1].set_xlabel("predicted price ($)")
    axes[1, 1].set_ylabel("residual ($)")
    axes[1, 1].set_title("B: residuals vs. prediction (heteroscedasticity check)")
    fig.suptitle("Residual diagnostics on the locked test set")
    savefig("04_residual_diagnostics.png")


def fig_bootstrap_distribution():
    preds = pd.read_csv(OUT / "test_predictions.csv")
    y = preds[TARGET].values
    pred_a = preds["pred_A"].values
    pred_b = preds["pred_B"].values
    rng = np.random.RandomState(42)
    n = len(y)
    n_boot = 5000
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        deltas[i] = rmse(y[idx], pred_a[idx]) - rmse(y[idx], pred_b[idx])
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(deltas, bins=80, color=COLOR_B, alpha=0.85)
    ax.axvline(0, color="k", linestyle="--", linewidth=1.5, label="H0: no difference")
    ax.axvline(ci_lo, color="gray", linestyle=":", linewidth=1.2)
    ax.axvline(ci_hi, color="gray", linestyle=":", linewidth=1.2, label="95% bootstrap CI")
    ax.set_xlabel("Bootstrap Δ RMSE = RMSE(A) − RMSE(B)  ($)")
    ax.set_ylabel("bootstrap resamples")
    ax.set_title("Paired bootstrap distribution of RMSE(A) − RMSE(B), n=5,000 resamples")
    ax.legend()
    savefig("05_bootstrap_delta_distribution.png")


def fig_ablation_bar():
    d = json.load(open(OUT / "ablation_cv.json"))
    order = [
        ("A_baseline_ols", "A: OLS\n(raw)"),
        ("B0_ridge_raw_features", "B0: Ridge\n(raw)"),
        ("B1a_ridge_plus_temporal_age", "B1a: +temporal/\nage"),
        ("B1b_ridge_plus_renovation", "B1b: +renovation"),
        ("B1c_ridge_plus_ratios", "B1c: +ratios"),
        ("B1d_ridge_plus_zipcode", "B1d: +zipcode"),
        ("B2_random_forest_engineered", "B2: Random\nForest"),
        ("B3_gradient_boosting_tuned", "B3: Tuned\nHGB (final)"),
    ]
    labels = [lab for k, lab in order]
    rmses = [d[k]["rmse"] for k, lab in order]
    stds = [d[k]["rmse_std"] for k, lab in order]
    colors = [COLOR_A] + [COLOR_B] * (len(order) - 1)

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(labels, rmses, yerr=stds, capsize=4, color=colors, alpha=0.85)
    ax.set_ylabel("5-fold CV RMSE ($)")
    ax.set_title("Ablation study: incremental CV RMSE by stage (error bars = fold std. dev.)")
    for bar, val in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 3000, f"{val:,.0f}",
                 ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=20, ha="right")
    savefig("06_ablation_cv_rmse.png")


def fig_permutation_importance():
    d = json.load(open(OUT / "importance.json"))
    ranking = d["ranking"][:12]
    names = [r["feature"] for r in ranking][::-1]
    vals = [r["importance_mean_rmse_increase"] for r in ranking][::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(names, vals, color=COLOR_B)
    ax.set_xlabel("mean RMSE increase (log-price space) when permuted")
    ax.set_title("Permutation importance, Proposed B, top 12 features")
    savefig("07_permutation_importance.png")


def fig_error_slices():
    slices = json.load(open(OUT / "error_slices.json"))
    names = [s["segment"] for s in slices]
    rmse_a = [s["rmse_A"] for s in slices]
    rmse_b = [s["rmse_B"] for s in slices]
    y_pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(y_pos - 0.2, rmse_a, height=0.4, color=COLOR_A, label="Baseline A")
    ax.barh(y_pos + 0.2, rmse_b, height=0.4, color=COLOR_B, label="Proposed B")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("RMSE ($)")
    ax.set_title("Error slicing: RMSE by subgroup, A vs. B (locked test set)")
    ax.legend()
    savefig("08_error_slices.png")


def fig_robustness_seeds():
    seeds = json.load(open(OUT / "robustness_seeds.json"))
    labels = [str(s["seed"]) for s in seeds]
    rmse_a = [s["rmse_A"] for s in seeds]
    rmse_b = [s["rmse_B"] for s in seeds]
    x = np.arange(len(labels))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x - width / 2, rmse_a, width, color=COLOR_A, label="Baseline A")
    axes[0].bar(x + width / 2, rmse_b, width, color=COLOR_B, label="Proposed B")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_xlabel("random seed (fresh dev/test split each time)")
    axes[0].set_ylabel("locked-test RMSE ($)")
    axes[0].set_title("A vs. B RMSE across 5 independent seeds")
    axes[0].legend()

    pct = [s["pct_improvement"] for s in seeds]
    axes[1].bar(labels, pct, color=COLOR_B)
    axes[1].axhline(0, color="k", linewidth=1)
    axes[1].set_xlabel("random seed")
    axes[1].set_ylabel("% RMSE reduction, B over A")
    axes[1].set_title("B's improvement over A is never close to zero")
    fig.suptitle("Robustness: B's advantage over A across 5 independent dev/test splits")
    savefig("09_robustness_seeds_A_vs_B.png")


def main():
    fig_eda_price_distribution()
    fig_eda_price_vs_sqft_and_grade()
    fig_actual_vs_predicted()
    fig_residuals()
    fig_bootstrap_distribution()
    fig_ablation_bar()
    fig_permutation_importance()
    fig_error_slices()
    fig_robustness_seeds()
    print(f"[make_plots] all figures written to {FIG}")


if __name__ == "__main__":
    main()
