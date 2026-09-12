"""
MarketProxy: a small, smooth model of "market conditions" over the calendar,
used only to fill in market_index / quote_signal when they are completely
unavailable for a given date (this happens for every row of
december-chart-inputs.csv, since that file has no such columns at all).

Why not just use the main gradient-boosted model for this?
Tree ensembles do not extrapolate trends: a date in November/December that
lies entirely outside the Jan-Oct training range gets bucketed with whatever
boundary values the trees saw, effectively freezing the "time" signal in a
way that doesn't need to be true. Since market_index shows a smooth,
plausibly-seasonal wave over the months (rising into May, falling into
September) plus a stable day-of-week pattern, we instead fit a *harmonic
regression* (a handful of sine/cosine terms) directly on the daily average.
Harmonic/Fourier terms extrapolate smoothly and periodically by
construction, which is a much safer assumption for a 31-day walk-forward
than either a flat carry-forward value or a tree's frozen boundary leaf.

This is intentionally simple (ordinary least squares, closed form) - the
point is smoothness and honest extrapolation, not squeezing out extra
accuracy on a signal that is mostly noise (quote_signal in particular has
very weak explanatory power - see the report).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class MarketProxy:
    def __init__(self, annual_harmonics: int = 2, weekly_harmonics: int = 1):
        self.annual_harmonics = annual_harmonics
        self.weekly_harmonics = weekly_harmonics
        self.coefs_: dict[str, np.ndarray] = {}
        self.means_: dict[str, float] = {}

    def _design_matrix(self, dates: pd.Series) -> np.ndarray:
        doy = dates.dt.dayofyear.to_numpy(dtype=float)
        dow = dates.dt.dayofweek.to_numpy(dtype=float)
        cols = [np.ones_like(doy)]
        for k in range(1, self.annual_harmonics + 1):
            cols.append(np.sin(2 * np.pi * k * doy / 365.25))
            cols.append(np.cos(2 * np.pi * k * doy / 365.25))
        for k in range(1, self.weekly_harmonics + 1):
            cols.append(np.sin(2 * np.pi * k * dow / 7.0))
            cols.append(np.cos(2 * np.pi * k * dow / 7.0))
        return np.column_stack(cols)

    def fit(self, daily: pd.DataFrame, columns: list[str]) -> "MarketProxy":
        """`daily` must have a 'date' column and one column per target,
        already aggregated to one row per calendar date (e.g. daily mean)."""
        X = self._design_matrix(daily["date"])
        for col in columns:
            y = daily[col].to_numpy(dtype=float)
            mask = ~np.isnan(y)
            coef, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
            self.coefs_[col] = coef
            self.means_[col] = float(np.nanmean(y))
        return self

    def predict(self, dates: pd.Series, column: str) -> np.ndarray:
        if column not in self.coefs_:
            raise KeyError(f"MarketProxy was not fit for column '{column}'")
        X = self._design_matrix(pd.to_datetime(dates))
        return X @ self.coefs_[column]

    def daily_curve(self, dates: pd.Series, column: str) -> pd.Series:
        return pd.Series(self.predict(dates, column), index=dates.index)


def fit_market_proxy(train_df: pd.DataFrame) -> MarketProxy:
    daily = (
        train_df.groupby(train_df["date"].dt.date)
        .agg(market_index=("market_index", "mean"), quote_signal=("quote_signal", "mean"))
        .reset_index()
        .rename(columns={"index": "date"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    proxy = MarketProxy(annual_harmonics=2, weekly_harmonics=1)
    proxy.fit(daily, ["market_index", "quote_signal"])
    return proxy
