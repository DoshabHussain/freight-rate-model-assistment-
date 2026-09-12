"""
Shared data loading, cleaning, and feature-engineering logic.

Used by both train.py and predict.py so that the exact same transformations
are applied at training time and at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_COLS = ["equipment"]
NUMERIC_FEATURE_COLS = [
    "log_distance",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "weight_abs",
    "weight_missing",
    "market_index_filled",
    "market_index_missing",
    "quote_signal_filled",
    "quote_signal_missing",
    "dow_sin",
    "dow_cos",
    "is_weekend",
]
ALL_FEATURE_COLS = NUMERIC_FEATURE_COLS + CATEGORICAL_COLS


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def build_city_coords(train_df: pd.DataFrame) -> dict:
    """Map city name -> (lat, lon), built once from train-test.csv.

    Needed because december-chart-inputs.csv only gives city *names*
    (Lexington, Fort Wayne) with no lat/lon columns at all. Since every city
    in that file also appears in train-test.csv with a single, consistent
    coordinate (verified during EDA - no city has more than one lat/lon),
    we can safely look coordinates up by name rather than require them as
    input.
    """
    pickups = train_df[["pickup", "pickup_lat", "pickup_lon"]].rename(
        columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
    )
    deliveries = train_df[["delivery", "delivery_lat", "delivery_lon"]].rename(
        columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
    )
    coords = pd.concat([pickups, deliveries]).drop_duplicates(subset="city").set_index("city")
    return {city: (row["lat"], row["lon"]) for city, row in coords.iterrows()}


def fill_missing_coords(df: pd.DataFrame, city_coords: dict) -> pd.DataFrame:
    """Fill pickup_lat/lon and delivery_lat/lon from a city lookup table
    whenever those columns are absent or null (e.g. december-chart-inputs.csv,
    which only provides city names)."""
    df = df.copy()
    for role in ["pickup", "delivery"]:
        lat_col, lon_col = f"{role}_lat", f"{role}_lon"
        if lat_col not in df.columns:
            df[lat_col] = np.nan
        if lon_col not in df.columns:
            df[lon_col] = np.nan
        needs_fill = df[lat_col].isna() | df[lon_col].isna()
        if needs_fill.any():
            looked_up = df.loc[needs_fill, role].map(city_coords)
            df.loc[needs_fill, lat_col] = looked_up.map(lambda t: t[0] if isinstance(t, tuple) else np.nan)
            df.loc[needs_fill, lon_col] = looked_up.map(lambda t: t[1] if isinstance(t, tuple) else np.nan)
    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues that don't require the market proxy model.

    Issue 1 - weight sign corruption: ~0.6% of weight values are negative,
    but their absolute-value distribution is statistically indistinguishable
    from the positive values (same mean/range). This looks like a sign-flip
    data entry error rather than a genuine measurement, so we take the
    absolute value and flag it so the model can still learn if the flag
    matters.

    Issue 2 - missing weight/market_index: both are legitimately missing
    (not corrupted) for a small fraction of rows. We flag missingness with
    an indicator column and impute a placeholder here; market_index gets a
    better, date-aware imputation later in `fill_market_features`.
    """
    df = df.copy()

    # --- date ---
    df["date"] = pd.to_datetime(df["date"])

    # --- weight ---
    df["weight_missing"] = df["weight"].isna().astype(int)
    df["weight_abs"] = df["weight"].abs()
    df["weight_abs"] = df["weight_abs"].fillna(df["weight_abs"].median())

    # --- distance ---
    df["log_distance"] = np.log(df["distance"].astype(float))

    # --- calendar features that are safe to extrapolate (recurring, bounded 0-6) ---
    dow = df["date"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    df["is_weekend"] = (dow >= 5).astype(int)

    return df


def fill_market_features(df: pd.DataFrame, market_proxy) -> pd.DataFrame:
    """Fill market_index / quote_signal.

    Strategy (in priority order):
      1. Use the real observed value if present.
      2. Otherwise use the mean of other rows sharing the same calendar date
         within this same dataframe (market_index/quote_signal vary mostly
         by date, only slightly by load - see EDA in the report).
      3. Otherwise (e.g. an entire date has no observed value at all, which
         is the case for december-chart-inputs.csv where the columns don't
         exist) fall back to the smooth `market_proxy` trend model fit on
         the training data, which extrapolates safely to unseen dates.
    """
    df = df.copy()
    for col in ["market_index", "quote_signal"]:
        if col not in df.columns:
            df[col] = np.nan

        df[f"{col}_missing"] = df[col].isna().astype(int)

        # step 2: within-file, same-date mean
        date_mean = df.groupby(df["date"].dt.date)[col].transform("mean")
        filled = df[col].fillna(date_mean)

        # step 3: proxy model trend, for any date with zero observed values
        still_missing = filled.isna()
        if still_missing.any():
            proxy_vals = market_proxy.predict(df.loc[still_missing, "date"], col)
            filled.loc[still_missing] = proxy_vals

        df[f"{col}_filled"] = filled

    return df


def build_features(df: pd.DataFrame, market_proxy, city_coords: dict | None = None) -> pd.DataFrame:
    df = basic_clean(df)
    if city_coords is not None:
        df = fill_missing_coords(df, city_coords)
    df = fill_market_features(df, market_proxy)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")
    return df


def get_X(df: pd.DataFrame) -> pd.DataFrame:
    return df[ALL_FEATURE_COLS]
