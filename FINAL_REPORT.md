# Final Report

Full narrative results are in `REPORT.md`; this file is the summary,
requirement traceability matrix, and final self-audit for the study.

**Revision note:** Task 1's implementation and report were revised after an
external technical review identified a real preprocessing-leakage bug and
several statistical-methodology and presentation issues. All of Task 1's
numbers below are from a fresh re-execution after those fixes, not edited
by hand. A second follow-up review caught 5 smaller remaining issues (a
literal-zero Wilcoxon p-value, an imprecise dependency-compatibility claim,
an imprecise "identical pipelines" statement, a missing threats-to-validity
discussion, and figure-reference completeness), all fixed in this version.
See `REPORT.md`'s "Changes from the first version" section for the full list
from both rounds.

## Executive summary

**Task 1** evaluated a transparent, reference-aligned OLS baseline (aligned
with the published-notebook family for this dataset, R²≈0.77) against an
incrementally-ablated proposed model (feature engineering — overwhelmingly
driven by zipcode encoding, per a feature-family ablation — plus a tuned
histogram gradient boosting model) for the "House Sales in King County,
USA" dataset. All feature engineering is now fit inside a proper
leakage-safe pipeline (bedroom-cap and ratio-imputation constants are
learned per-CV-fold, never from data outside the training partition).
Result: B reduces RMSE by 51.6% (232,823→112,588 dollars) on the locked
test set, with a paired permutation-test p-value < 1/5,001 (the smallest
value a 5,000-iteration test can resolve) and a 95% bootstrap effect-size
CI of [63,339, 200,530], and raises R² from 0.569 to 0.899. The improvement
is stable across 5 independent dev/test splits, ranging from 34.7% to
53.1% RMSE reduction in B's favor every time — never small, never
reversed.

## Requirement Traceability Matrix

### Task 1

| Req ID | Original Requirement | Interpretation | Implementation | Validation | Artifact | Status |
|---|---|---|---|---|---|---|
| T1-1 | Select Kaggle dataset with real baseline | Dataset must be verifiably real; baseline must be a real, cited notebook | King County house sales (harlfoxem); reference = kyryllvasylenko OLS notebook | Web search confirmed dataset/notebook existence and title-reported score; full notebook body unreachable (Kaggle SPA) — disclosed, not hidden; language changed from "reproduction" to "reference-aligned baseline" after review | `task1/REPORT.md` §1-2 | DONE (with disclosed verification limit) |
| T1-2 | Data audit before modeling | Structured audit of schema, missing values, duplicates, leakage, outliers | `src/data.py::audit` | Executed against real data; JSON output | `task1/outputs/audit.json` | DONE |
| T1-3 | Leakage-safe preprocessing | Fit only on train; document leakage paths | `HouseFeatureEngineer` sklearn transformer (fits bedroom-cap/ratio medians per-fold), inside every model `Pipeline`; group-aware split | Assertion that dev/test id sets are disjoint; split now happens BEFORE any feature engineering (post-review fix — see item below) | `src/features.py`, `src/run_experiment.py` | DONE (leakage bug found and fixed) |
| T1-4 | One evaluation protocol, same folds for A/B | Deterministic seeds, locked test set | Group split + 5-fold `GroupKFold`, seed=42 | Same fold indices reused across every stage | `src/evaluation.py` | DONE |
| T1-5 | Reference-align baseline, document deviations | OLS on raw numeric features; exact original notebook code unverifiable, disclosed | `models.py::baseline_model` | CV + locked-test metrics computed | `outputs/ablation_cv.json`, `outputs/final_ab_results.json` | DONE |
| T1-6 | Build proposed model B, avoid needless complexity | Incremental ablation B0→B3, now with B1 decomposed into B1a-B1d | `models.py`, `run_experiment.py` | Each stage's CV RMSE measured | `outputs/ablation_cv.json` | DONE |
| T1-7 | Ablation study | Version/change/metric/delta table | B0 (algorithm only) → B1a-B1d (feature families added one at a time) → B2 (+trees) → B3 (+tuning) | Real CV numbers per stage; family-level attribution now evidenced, not asserted | `task1/REPORT.md` §5 | DONE (extended past original scope per review) |
| T1-8 | Statistical A/B test, H0/H1, justified test choice | Paired bootstrap CI (effect size) + paired permutation test (H0) + Wilcoxon on paired absolute error (secondary) | `evaluation.py::full_ab_report` | 5,000-resample bootstrap + 5,000-iteration permutation test + Wilcoxon, all computed, cleanly separated per metric | `outputs/final_ab_results.json` | DONE (methodology corrected post-review) |
| T1-9 | Distinguish statistical vs practical significance | Explicit discussion | REPORT.md §6 | CI-excludes-zero + permutation p-value + effect-size-in-dollars framing | `task1/REPORT.md` §6 | DONE |
| T1-10 | Bootstrap analysis, visualize | Paired bootstrap distribution + CI, now plotted | `paired_bootstrap_ci`, `make_plots.py::fig_bootstrap_distribution` | 5,000 resamples computed and rendered as a histogram | `outputs/figures/05_bootstrap_delta_distribution.png` | DONE (plot added post-review) |
| T1-11 | Model diagnostics | Residuals, error distribution, now plotted | `diagnostics.py`, `make_plots.py` | Residual mean/std/skew, %-within-10%, heteroscedasticity check + actual-vs-predicted and residual plots | `outputs/diagnostics.json`, `outputs/figures/03_*.png`, `04_*.png` | DONE |
| T1-12 | Error slicing by subgroup | Segment table, no causal claims, now plotted | grade/waterfront/age/price/condition bins | 16 segments computed on real predictions | `outputs/error_slices.json`, `outputs/figures/08_error_slices.png` | DONE |
| T1-13 | Interpretability | Feature importance, no causal claims, explicitly scoped relative to the ablation | Permutation importance (chosen over SHAP to avoid an extra heavy dependency) | Computed on held-out subsample; report now explicitly notes this ranking does not "explain" the ablation's B0→B1 gain (that evidence comes from §5's family decomposition instead) | `outputs/importance.json`, `outputs/figures/07_*.png` | DONE (SHAP substitution disclosed; scope-confusion fixed post-review) |
| T1-14 | Robustness / sensitivity | A vs B compared across seeds, not B alone | 5 seeds, full re-split + refit of BOTH models + locked-test comparison | Real re-execution; delta and % improvement reported per seed | `outputs/robustness_seeds.json`, `outputs/figures/09_*.png` | DONE (upgraded post-review from "B alone" to "A vs B") |
| T1-15 | Computational cost | Train/inference time | `time.perf_counter` around fit/predict | Measured, reported | `outputs/final_ab_results.json` | DONE |
| T1-16 | Final results table + honest conclusion | No forced positive result | REPORT.md §6, §11 | — | `task1/REPORT.md` | DONE |
| T1-17 | Hyperparameter search aligned with primary metric | Search should optimize the same metric the final report headlines | Custom dollar-space RMSE scorer (`models.py::dollar_rmse_scorer`) replaces the log-space default scorer | Search re-run and re-verified with the new scorer | `src/models.py`, `outputs/ablation_cv.json` | DONE (added post-review; previously an undisclosed metric-space mismatch) |
| T1-18 | Reproducibility: pinned versions, recorded environment | Exact installed versions, not guessed; the environment is recorded programmatically | `requirements.txt` pins the modern scientific stack used by this study; `run_experiment.py` records `platform`/`numpy`/`pandas`/`sklearn`/`scipy` versions programmatically | Environment file written every run | `requirements.txt`, `outputs/environment.json` | DONE |
| T1-19 | No p-value should be reported as a literal, unqualified 0 | Distinguish a genuine floating-point underflow from a claim of exact zero probability | `evaluation.py::wilcoxon_on_paired_absolute_errors` now returns `p_value_raw`, `p_value_underflowed`, and `p_value_report` ("< machine precision") instead of a bare `p_value: 0.0` | Verified the underflow is genuine (not a bug) via an independent hand-computation of the normal-approximation z-score and its log-survival-function | `outputs/final_ab_results.json`, `task1/REPORT.md` §6 | DONE (added after a follow-up review) |

## Final Quality Gate — self-audit against the assignment's checklist

- [x] Real Kaggle dataset used — verified via a byte-identical public mirror (Kaggle's own download API/UI was unreachable from this sandboxed network; documented in `task1/REPORT.md` §1).
- [~] Real published Kaggle baseline verified — notebook's existence, author, and title-reported score (R²=0.77) verified via search; full notebook body could not be scraped (Kaggle SPA rendering). Our implementation is now explicitly labeled a "reference-aligned baseline," not a "reproduction," to match what was actually verified.
- [x] Fair A/B comparison — identical development/test observations, identical CV fold assignments, identical target definition and transform, and preprocessing fit exclusively within training partitions for both models. Model-specific feature pipelines are intentionally *different* (A: raw numeric columns; B: engineered features) because feature engineering is part of Treatment B, not a nuisance variable to be held constant — what is held constant is the evaluation protocol, not the treatment.
- [x] Same evaluation observations — both models scored on the same 6,485-row locked test set.
- [x] Leakage prevented — group-aware split by house id performed BEFORE any feature engineering; bedroom-cap and ratio-imputation constants fit inside a proper `sklearn` transformer, per-CV-fold, never from data outside the training partition. (A leakage bug in the pre-review version — the bedroom cap was fit on the full pre-split dataset — was found and fixed; see `task1/REPORT.md` "Changes from the first version".)
- [x] Primary metric defined before final-test evaluation — RMSE/MAE/R² fixed in `evaluation.py` before `run_experiment.py`'s final section ever touches `test_df`; hyperparameter search now also optimizes this same dollar-space metric, not a different log-space proxy.
- [x] Confidence intervals calculated — 95% paired-bootstrap CIs for RMSE, MAE, R², explicitly framed as effect-size intervals, not hypothesis tests.
- [x] Statistical test justified — paired permutation test (matches the paired-observation design, makes no distributional assumption) as the primary H0 test per metric, plus Wilcoxon signed-rank on paired absolute errors as a secondary, distinct check. No p-value is reported as a literal 0; both Monte Carlo and asymptotic-underflow cases report a verified upper bound instead.
- [x] Statistical vs practical significance distinguished — explicit discussion in `task1/REPORT.md` §6.
- [x] Ablation study completed — B0 through B3, with B1 decomposed into B1a-B1d feature-family stages so the largest single gain (zipcode encoding) is evidenced, not asserted.
- [x] Error analysis completed — 16 subgroup slices, real numbers, now plotted.
- [x] Model diagnostics completed — residual mean/std/skew, heteroscedasticity check, worst-case errors, now plotted (actual-vs-predicted, residual histograms, residuals-vs-prediction).
- [x] Unit tests pass — 8/8 targeted tests covering the leakage-safe transformer, group split, p-value correction, and paired-metric validation.
- [x] README contains reproducible commands and pinned dependencies.
- [x] No p-value reported as a literal, unqualified 0 — the Wilcoxon test's underflowed `0.0` is now reported with an explicit `p_value_underflowed` flag and a `p_value_report: "< machine precision"` string, with the underflow independently verified (not assumed) via a hand-computed z-score check.
- [x] Threats to Validity documented — `task1/REPORT.md` §12 covers external validity, baseline fidelity, split sensitivity, hyperparameter-search scope, interpretability scope, and the statistical-independence assumption.
- [x] Every figure referenced in the report — verified programmatically (all 9 filenames in `outputs/figures/` appear in `task1/REPORT.md`).
- [x] No fabricated numbers/results — every number in both reports was produced by a fresh execution captured in this session; the one intentionally-flagged exception is the baseline notebook's own self-reported R²=0.77, quoted as *their* number, not computed by us.
- [x] Figures included — 9 figures covering EDA, actual-vs-predicted, residuals, the bootstrap distribution, the ablation bar chart, permutation importance, error slices, and seed robustness (`task1/outputs/figures/`), added after the review flagged their absence as the weakest part of the original submission.

### Known deviations (disclosed, not hidden)

1. **Baseline notebook internals** could not be scraped (Kaggle SPA rendering) — the implementation is explicitly labeled a "reference-aligned baseline" rather than a "reproduction" for exactly this reason. See T1-1, T1-5.
2. **SHAP was not used**; permutation importance was used instead, to avoid adding a heavy dependency (both are model-agnostic, post-hoc importance measures; SHAP's additional value — additive per-prediction attribution — was not required by the specific interpretability questions asked). The report now explicitly scopes this ranking to the final model B and does not use it to explain the ablation's feature-family gain (that evidence comes from the ablation itself). See T1-13.
3. **Nested cross-validation** was not used for the final A/B comparison, by design: the final comparison is a single locked hold-out (not a CV estimate), so nesting does not apply there; hyperparameter search for B3 used its own 3-fold (not 5-fold) GroupKFold purely to bound wall-clock time in this sandboxed environment, with the winning configuration re-validated under the full 5-fold protocol before being compared to anything else.
4. **The project has a small, targeted unit-test suite** (`tests/test_task1.py`, 8 tests, added after a second follow-up review) covering exactly the design claims where a subtle bug would be dangerous and easy to miss by inspection: that `HouseFeatureEngineer.transform()` genuinely does not learn anything from the data it's transforming (only from `.fit()`), that the group-aware split has zero house-id overlap and is seed-deterministic, that the Monte Carlo p-value correction never returns an unqualified zero (and behaves sensibly at the 0%/50%/100%-exceedance boundaries), and that the paired A/B metric functions fail loudly rather than silently misaligning mismatched-length predictions. The broader correctness argument rests primarily on the leakage assertion in `run_experiment.py`, the disjoint-id check, and full re-execution producing internally consistent numbers across the ablation, final test, diagnostics, and robustness artifacts.
