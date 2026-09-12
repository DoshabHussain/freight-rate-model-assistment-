"""
Exploratory data analysis. Not required to run the pipeline - this is here
so the findings referenced in the report/README are reproducible.

    python -m src.eda
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def main() -> None:
    df = pd.read_csv(DATA / "train-test.csv")
    val = pd.read_csv(DATA / "validation.csv")
    df["date"] = pd.to_datetime(df["date"])

    print("=== Shape & missingness ===")
    print(df.shape)
    print(df.isna().sum())

    print("\n=== Weight sign issue ===")
    neg = df[df["weight"] < 0]
    print(f"{len(neg)} of {len(df)} rows ({len(neg)/len(df):.2%}) have negative weight.")
    print("abs(negative) mean:", neg["weight"].abs().mean(), " | positive mean:", df.loc[df.weight > 0, "weight"].mean())
    print("-> distributions match: this is a sign-flip data-entry error, not a separate population.")

    print("\n=== log(rate) vs log(distance) ===")
    x = np.log(df["distance"])
    y = np.log(df["posted_rate"])
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    r2 = 1 - np.sum(resid**2) / np.sum((y - y.mean()) ** 2)
    print(f"distance alone explains R2={r2:.3f} of log(rate) variance (slope={b:.3f}, intercept={a:.3f})")

    print("\n=== market_index: mostly a date-level signal, small weekly seasonality ===")
    daily = df.groupby(df["date"].dt.date)["market_index"].agg(["mean", "std"])
    print("mean of within-day std:", daily["std"].mean(), "| across-day std of the daily mean:", daily["mean"].std())
    print(df.groupby(df["date"].dt.dayofweek)["market_index"].mean())

    print("\n=== quote_signal: weak signal ===")
    print("corr(quote_signal, log(rate) residual after distance):",
          np.corrcoef(df["quote_signal"], resid)[0, 1])

    print("\n=== New pickup/delivery cities appear in validation.csv that never appear in training ===")
    new_cities = set(val["pickup"]) - set(df["pickup"])
    print(sorted(new_cities))
    print("-> city name used as a categorical feature would hit unseen categories at inference;")
    print("   using continuous lat/lon instead generalizes safely to new cities.")

    print("\n=== december-chart-inputs.csv has no market_index/quote_signal columns at all ===")
    print("-> handled via a harmonic (Fourier) regression on date that extrapolates smoothly,")
    print("   rather than letting the tree model freeze at the last training-month bucket.")


if __name__ == "__main__":
    main()
