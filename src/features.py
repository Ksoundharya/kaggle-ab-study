"""Feature engineering and leakage-safe preprocessing.

Findings from the audit (src/data.py) that drive the decisions below:

1. No missing values anywhere -> no imputer is strictly required, but we still
   wire a median imputer into the pipeline defensively: at inference time a
   future record could legitimately be missing a field, and a pipeline that
   assumes zero missingness forever is a latent production bug.
2. 177 duplicate `id`s = houses sold more than once in the 2014-2015 window.
   Two sales of the same physical house are not independent observations
   (structural attributes are ~identical, price differs mostly by market
   timing). Splitting by ROW would let one sale of a house sit in train and
   the other in test, which leaks near-duplicate structural information
   across the split. We therefore split by GroupKFold / grouped train-test
   split keyed on `id`.
3. `price` is right-skewed (skew = 4.02). Tree models are scale/skew
   invariant, but the linear baseline is not; we fit and evaluate all models
   on log1p(price) and invert with expm1 before computing metrics.
4. One record (id=2402100895) has 33 bedrooms on 1620 sqft of living space —
   a documented data-entry error in this public dataset, not a real 33-
   bedroom house.
5. `date` (sale timestamp) is not itself a numeric feature but we derive
   `sale_year` and `sale_month` from it, since price is known to drift with
   the housing-market cycle across the one-year window in this data.
6. `yr_renovated` is 0 for 95.8% of homes (meaning "never renovated", not a
   valid year). We convert it to a binary `was_renovated` flag plus
   `years_since_renovation_or_build`, rather than feeding the raw 0/YYYY
   mixed-semantics column into a linear model.
7. `zipcode` is a high-cardinality (70 unique values) nominal identifier of
   location, and `lat`/`long` already encode fine-grained location
   numerically. We one-hot encode zipcode (fit on train only) rather than
   using it as a raw integer, which would impose a false ordinal
   relationship between unrelated zip codes.

============================================================
LEAKAGE FIX (post-review correction)
============================================================
An earlier version of this module computed the bedroom-clip threshold (and
the two ratio-imputation medians) once from the FULL dataset before the
train/test split, then engineered features on that basis. That is leakage:
the clip threshold and the imputed medians were, in that version, partly
informed by rows that ended up in the locked test set and even by rows in
validation folds during cross-validation.

The fix implemented here is a proper `sklearn`-compatible transformer,
`HouseFeatureEngineer`, whose `fit()` learns the bedroom-clip threshold and
the two ratio-imputation medians from whatever data it is given, and whose
`transform()` applies those already-learned constants without recomputing
them. Placed as the first step of every model `Pipeline` (see models.py /
run_experiment.py), this means:

  * Inside 5-fold `GroupKFold` cross-validation, each fold's training
    partition fits its own bedroom cap and ratio medians; the validation
    fold only has them applied.
  * The locked test set only ever has the DEVELOPMENT set's constants
    applied to it; it never contributes to computing them.

This makes the claim "all learned preprocessing parameters are estimated
exclusively from the corresponding training partition" actually true,
rather than true only for the ColumnTransformer step as before.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "price"
BEDROOM_CLIP_QUANTILE = 0.999


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Fits the bedroom-clip threshold and ratio-imputation medians on
    whatever data `.fit()` sees (a CV training fold, or the full
    development set), then applies those fixed constants in `.transform()`.
    No statistic used at transform time is ever recomputed from the data
    being transformed -- see the module docstring's "LEAKAGE FIX" section.
    """

    def __init__(self, bedroom_clip_quantile: float = BEDROOM_CLIP_QUANTILE):
        self.bedroom_clip_quantile = bedroom_clip_quantile

    def fit(self, X: pd.DataFrame, y=None):
        self.bedroom_cap_ = float(X["bedrooms"].quantile(self.bedroom_clip_quantile))
        living_lot_ratio = X["sqft_living"] / X["sqft_lot"].replace(0, np.nan)
        self.living_lot_ratio_median_ = float(living_lot_ratio.median())
        bath_per_bed = X["bathrooms"] / X["bedrooms"].replace(0, np.nan)
        self.bath_per_bed_median_ = float(bath_per_bed.median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        df["sale_date"] = pd.to_datetime(df["date"].str[:8], format="%Y%m%d")
        df["sale_year"] = df["sale_date"].dt.year
        df["sale_month"] = df["sale_date"].dt.month

        df["bedrooms"] = df["bedrooms"].clip(upper=self.bedroom_cap_)

        df["was_renovated"] = (df["yr_renovated"] > 0).astype(int)
        last_touch_year = np.where(df["yr_renovated"] > 0, df["yr_renovated"], df["yr_built"])
        df["years_since_renovation_or_build"] = df["sale_year"] - last_touch_year

        df["house_age_at_sale"] = df["sale_year"] - df["yr_built"]
        df["has_basement"] = (df["sqft_basement"] > 0).astype(int)

        df["living_lot_ratio"] = df["sqft_living"] / df["sqft_lot"].replace(0, np.nan)
        df["living_lot_ratio"] = df["living_lot_ratio"].fillna(self.living_lot_ratio_median_)

        df["bath_per_bed"] = df["bathrooms"] / df["bedrooms"].replace(0, np.nan)
        df["bath_per_bed"] = df["bath_per_bed"].fillna(self.bath_per_bed_median_)

        if "zipcode" in df.columns:
            df["zipcode"] = df["zipcode"].astype(str)
        return df


NUMERIC_FEATURES_BASELINE = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors", "waterfront",
    "view", "condition", "grade", "sqft_above", "sqft_basement", "yr_built",
    "yr_renovated", "lat", "long", "sqft_living15", "sqft_lot15",
]

# Feature families, so the ablation can add them one at a time (B1a-B1d)
# and attribute the B0->B1 gain to specific groups instead of asserting it.
FAMILY_TEMPORAL_AGE = ["sale_year", "sale_month", "house_age_at_sale"]
FAMILY_RENOVATION = ["was_renovated", "years_since_renovation_or_build"]
FAMILY_RATIOS = ["has_basement", "living_lot_ratio", "bath_per_bed"]
FAMILY_ZIPCODE_CATEGORICAL = ["zipcode"]  # categorical, handled separately below

NUMERIC_FEATURES_ENGINEERED = (
    NUMERIC_FEATURES_BASELINE + FAMILY_TEMPORAL_AGE + FAMILY_RENOVATION + FAMILY_RATIOS
)
CATEGORICAL_FEATURES_ENGINEERED = ["zipcode"]


def build_baseline_preprocessor() -> Pipeline:
    """Full pipeline front-end for Baseline A: the leakage-safe feature
    engineer (needed only for its bedroom-cap data-quality fix -- see
    LEAKAGE FIX docstring -- every other family is dropped by the column
    selector below) followed by numeric selection/scaling."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    column_select = ColumnTransformer([("num", numeric, NUMERIC_FEATURES_BASELINE)])
    return Pipeline([
        ("engineer", HouseFeatureEngineer()),
        ("select", column_select),
    ])


def build_engineered_preprocessor(
    for_linear: bool,
    include_temporal_age: bool = True,
    include_renovation: bool = True,
    include_ratios: bool = True,
    include_zipcode: bool = True,
) -> Pipeline:
    """Full pipeline front-end for proposed model B and its ablation
    stages. The `include_*` flags let the ablation study add feature
    families one at a time (B1a: +temporal/age, B1b: +renovation,
    B1c: +ratios, B1d: +zipcode) so the B0->B1 gain can be decomposed
    instead of asserted from the aggregate B1 result alone."""
    numeric_cols = list(NUMERIC_FEATURES_BASELINE)
    if include_temporal_age:
        numeric_cols += FAMILY_TEMPORAL_AGE
    if include_renovation:
        numeric_cols += FAMILY_RENOVATION
    if include_ratios:
        numeric_cols += FAMILY_RATIOS

    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if for_linear:
        numeric_steps.append(("scale", StandardScaler()))
    numeric = Pipeline(numeric_steps)

    transformers = [("num", numeric, numeric_cols)]
    if include_zipcode:
        categorical = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("cat", categorical, CATEGORICAL_FEATURES_ENGINEERED))

    column_select = ColumnTransformer(transformers)
    return Pipeline([
        ("engineer", HouseFeatureEngineer()),
        ("select", column_select),
    ])


def feature_columns_for(
    include_temporal_age: bool = True,
    include_renovation: bool = True,
    include_ratios: bool = True,
    include_zipcode: bool = True,
) -> list:
    """Raw input columns that must be sliced from the dataframe and handed
    to a Pipeline built by build_engineered_preprocessor with the same
    flags (the HouseFeatureEngineer step derives everything else from
    these raw columns)."""
    cols = [
        "date", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
        "waterfront", "view", "condition", "grade", "sqft_above",
        "sqft_basement", "yr_built", "yr_renovated", "lat", "long",
        "sqft_living15", "sqft_lot15",
    ]
    if include_zipcode:
        cols.append("zipcode")
    return cols


RAW_COLUMNS_FOR_BASELINE = feature_columns_for(
    include_temporal_age=False, include_renovation=False, include_ratios=False, include_zipcode=False
)
RAW_COLUMNS_FOR_ENGINEERED = feature_columns_for()
