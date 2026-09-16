"""Data loading and the initial audit for the King County house sales dataset.

Dataset: "House Sales in King County, USA" (Kaggle: harlfoxem/housesalesprediction).
21,613 single-family home sales in King County, WA, May 2014 - May 2015.
Source file reproduced from the public mirror at
https://raw.githubusercontent.com/Shreyas3108/house-price-prediction/master/kc_house_data.csv
(schema and row count match the Kaggle dataset exactly: 21 columns, 21,613 rows,
0 missing values, same column names/units as harlfoxem's original upload).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "kc_house_data.csv"
TARGET = "price"

# Columns that are pure identifiers or that would leak post-hoc information
# (id is a database key with no predictive content; date is the transaction
# timestamp, kept only to derive a sale-year/season feature, not used raw).
ID_COLS = ["id"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@dataclass
class AuditResult:
    n_rows: int
    n_cols: int
    n_duplicates: int
    n_duplicate_ids: int
    missing_by_col: Dict[str, int]
    dtypes: Dict[str, str]
    target_describe: Dict[str, float]
    target_skew: float
    zero_or_negative_price_rows: int
    bedrooms_max: int
    bedrooms_zero_rows: int
    bathrooms_zero_rows: int
    sqft_living_min: int
    yr_renovated_zero_share: float
    waterfront_positive_share: float
    suspicious_bedroom_outlier_ids: list


def audit(df: pd.DataFrame) -> AuditResult:
    """Structured data audit executed BEFORE any modeling decision.

    Findings recorded here directly drive the preprocessing choices in
    features.py (see docstring there for the mapping from finding -> action).
    """
    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df["id"].duplicated().sum())
    missing = df.isna().sum().to_dict()
    missing = {k: int(v) for k, v in missing.items()}
    dtypes = {k: str(v) for k, v in df.dtypes.items()}

    target_desc = df[TARGET].describe().to_dict()
    target_skew = float(df[TARGET].skew())

    zero_neg_price = int((df[TARGET] <= 0).sum())
    bedrooms_max = int(df["bedrooms"].max())
    bedrooms_zero = int((df["bedrooms"] == 0).sum())
    bathrooms_zero = int((df["bathrooms"] == 0).sum())
    sqft_living_min = int(df["sqft_living"].min())
    yr_renov_zero_share = float((df["yr_renovated"] == 0).mean())
    waterfront_share = float((df["waterfront"] == 1).mean())

    # The well-known 33-bedroom / 1.75-bathroom / 1620 sqft record is a
    # documented data-entry error in this public dataset (physically
    # implausible bedroom-to-floor-area ratio). Flag by id rather than
    # silently dropping so the decision is auditable.
    suspicious = df.loc[
        (df["bedrooms"] >= 10) & (df["sqft_living"] < df["bedrooms"] * 200),
        "id",
    ].tolist()

    return AuditResult(
        n_rows=len(df),
        n_cols=df.shape[1],
        n_duplicates=dup_rows,
        n_duplicate_ids=dup_ids,
        missing_by_col=missing,
        dtypes=dtypes,
        target_describe={k: float(v) for k, v in target_desc.items()},
        target_skew=target_skew,
        zero_or_negative_price_rows=zero_neg_price,
        bedrooms_max=bedrooms_max,
        bedrooms_zero_rows=bedrooms_zero,
        bathrooms_zero_rows=bathrooms_zero,
        sqft_living_min=sqft_living_min,
        yr_renovated_zero_share=yr_renov_zero_share,
        waterfront_positive_share=waterfront_share,
        suspicious_bedroom_outlier_ids=[int(x) for x in suspicious],
    )


def save_audit(result: AuditResult, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2)


if __name__ == "__main__":
    df = load_raw()
    result = audit(df)
    save_audit(result, Path(__file__).resolve().parents[1] / "outputs" / "audit.json")
    print(json.dumps(asdict(result), indent=2)[:2000])
