# Take-Home Submission: Kaggle A/B Study (Task 1) + Flight Data Engineering (Task 2)

See `FINAL_REPORT.md` for the full technical report, requirement
traceability matrix, and final quality-gate self-audit.

```
project/
├── README.md                  (this file)
├── FINAL_REPORT.md             full report + traceability matrix + audit
├── requirements.txt            index only -- see task1/ and task2/ for the real, per-task files
├── task1/
│   ├── REPORT.md               Task 1 findings, all numbers real-executed
│   ├── requirements.txt        pinned scientific-Python stack, Python 3.11.15 (Task 1 has no 3.7 requirement)
│   ├── data/kc_house_data.csv  the dataset used (see REPORT.md §1 for provenance)
│   ├── src/
│   │   ├── data.py             loading + audit
│   │   ├── features.py         feature engineering + leakage-safe preprocessing
│   │   ├── models.py           baseline A + ablation stages B0-B3
│   │   ├── evaluation.py       split protocol, metrics, paired bootstrap, Wilcoxon
│   │   ├── run_experiment.py   end-to-end: audit -> CV -> ablation -> locked test A/B
│   │   ├── diagnostics.py      residuals, error slicing, permutation importance, A-vs-B seed robustness
│   │   └── make_plots.py       generates all 9 figures referenced in REPORT.md
│   ├── tests/test_task1.py     8 targeted unit tests (leakage, split, p-value correction, paired metrics)
│   └── outputs/                every JSON artifact + outputs/figures/*.png the report pulls from
└── task2/
    ├── README.md                Task 2 findings, filename-collision analysis, all numbers real-executed
    ├── requirements.txt         Python 3.7+, stdlib-only (pytest is a dev-only dependency)
    ├── config.py
    ├── generate_flights.py      Phase 1: data generation
    ├── analyze_flights.py       Phase 2: streaming processing + analytics
    ├── models.py                record validation (dirty vs invalid)
    ├── statistics.py            dependency-free descriptive/inferential stats
    ├── outputs/analysis_report.json
    └── tests/
        ├── test_generator.py
        └── test_analysis.py
```

## Reproduction

**Task 1** was executed and validated on Python 3.11.15 with the pinned
scientific-Python stack in `task1/requirements.txt` (exact versions
recorded programmatically in `task1/outputs/environment.json` on every
run). Task 1 has no Python-3.7 requirement in the assignment, so its
dependencies are pinned to modern versions rather than constrained to 3.7.
```bash
cd task1
pip install -r requirements.txt
python3 src/run_experiment.py     # ~80s: audit, CV ablation (A, B0, B1a-B1d, B2, B3), locked-test A/B test
python3 src/diagnostics.py        # ~10s: residuals, error slicing, importance, A-vs-B seed robustness
python3 src/make_plots.py         # ~5s: writes outputs/figures/*.png
pytest tests/ -v                  # ~1s: 8 targeted unit tests
```

**Task 2** is intentionally implemented with zero third-party runtime
dependencies, using only Python 3.7+ standard-library functionality, to
satisfy the assignment's explicit Python 3.7+ requirement (see
`task2/requirements.txt`; `pytest` there is dev/test-only).
```bash
cd task2
python3 generate_flights.py --output-dir /tmp/flights --num-files 5000 --seed 42
python3 analyze_flights.py --input-dir /tmp/flights --output-report outputs/analysis_report.json
pip install -r requirements.txt   # pytest only, for running the test suite
pytest tests/ -v
```
