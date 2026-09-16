# Kaggle A/B Study: King County House Prices

See `FINAL_REPORT.md` for the full technical report, requirement
traceability matrix, and final quality-gate self-audit.

## Project Overview

This project studies the **King County House Price Prediction** problem by
comparing a reference Kaggle Ordinary Least Squares baseline with an improved
Gradient Boosting model. Rather than only comparing model scores, the
experiment prevents data leakage, keeps a separate locked test set, uses the
same data splits for both models, and evaluates each improvement incrementally
through an ablation study.

The analysis uses paired bootstrap confidence intervals, permutation testing,
Wilcoxon testing, error analysis, residual diagnostics, feature importance,
and multiple random-seed experiments to assess whether the improvement is
consistent and statistically meaningful. The project also includes
visualizations, automated tests, fixed random seeds, and pinned dependencies
so the results can be reproduced and verified.

```
project/
├── README.md                  (this file)
├── FINAL_REPORT.md             full report + traceability matrix + audit
├── requirements.txt            pinned scientific-Python stack
├── REPORT.md                   findings and methodology
├── data/kc_house_data.csv      the dataset used
├── src/                        experiment implementation
├── tests/test_task1.py         targeted unit tests
└── outputs/                    JSON artifacts and report figures
```

## Reproduction

The project was executed and validated on Python 3.11.15 with the pinned
scientific-Python stack in `task1/requirements.txt` (exact versions
recorded programmatically in `task1/outputs/environment.json` on every
run).
```bash
cd task1
pip install -r requirements.txt
python3 src/run_experiment.py     # ~80s: audit, CV ablation (A, B0, B1a-B1d, B2, B3), locked-test A/B test
python3 src/diagnostics.py        # ~10s: residuals, error slicing, importance, A-vs-B seed robustness
python3 src/make_plots.py         # ~5s: writes outputs/figures/*.png
pytest tests/ -v                  # ~1s: 8 targeted unit tests
```

