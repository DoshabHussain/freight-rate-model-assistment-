"""
Train the freight rate model.

Usage:
    python -m src.train

Produces:
    models/rate_model.json      - trained XGBoost booster
    models/market_proxy.pkl     - fitted MarketProxy (for date extrapolation)
    models/feature_cols.json    - exact feature list/order used at inference

Validation strategy
--------------------
The task is inherently a forecasting problem: we train on Jan-Oct 2025 and
must predict Nov-Dec 2025, i.e. dates strictly *after* anything the model
has seen. A random/shuffled train-test split would let the model "peek" at
market conditions from days surrounding a hidden day, which overstates how
well it will generalize to a genuinely future month. So:

  1. Primary check: a single chronological split - train on the first ~85%
     of dates, hold out the most recent ~15% of dates as validation. This
     mirrors the real deployment scenario (predict the future from the
     past) and is the number we trust most.
  2. Robustness check: 5-fold expanding-window TimeSeriesSplit, to make
     sure the holdout result isn't a fluke of exactly where the cut falls.
  3. Final model: once the approach is validated, we refit on 100% of
     train-test.csv (no holdout) so the shipped model uses all available
     signal before predicting validation.csv / december-chart-inputs.csv.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

try:
    from src.data_prep import ALL_FEATURE_COLS, basic_clean, build_city_coords, fill_market_features
    from src.market_proxy import fit_market_proxy
except ImportError:
    from data_prep import ALL_FEATURE_COLS, basic_clean, build_city_coords, fill_market_features
    from market_proxy import fit_market_proxy

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=5,
    learning_rate=0.04,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.5,
    min_child_weight=5,
    objective="reg:squarederror",
    enable_categorical=True,
    tree_method="hist",
    random_state=42,
)


def evaluate(y_true_log: np.ndarray, y_pred_log: np.ndarray, label: str) -> dict:
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    metrics = {
        "mae_$": mean_absolute_error(y_true, y_pred),
        "rmse_$": mean_squared_error(y_true, y_pred) ** 0.5,
        "mape_%": mean_absolute_percentage_error(y_true, y_pred) * 100,
        "r2": r2_score(y_true, y_pred),
    }
    print(f"[{label}] " + "  ".join(f"{k}={v:,.3f}" for k, v in metrics.items()))
    return metrics


def main() -> None:
    raw = pd.read_csv(DATA / "train-test.csv")
    raw = basic_clean(raw)
    raw = raw.sort_values("date").reset_index(drop=True)

    # Fit the market-condition proxy on the *raw* observed market_index /
    # quote_signal (before any imputation), so it learns the true signal.
    market_proxy = fit_market_proxy(raw)
    city_coords = build_city_coords(raw)

    df = fill_market_features(raw, market_proxy)
    for col in ["equipment"]:
        df[col] = df[col].astype("category")

    y = np.log(df["posted_rate"].to_numpy(dtype=float))
    X = df[ALL_FEATURE_COLS]

    # ---------- 1. chronological holdout ----------
    cut = int(len(df) * 0.85)
    X_train, X_hold = X.iloc[:cut], X.iloc[cut:]
    y_train, y_hold = y[:cut], y[cut:]
    print(f"Chronological split: train={len(X_train)} rows ({df['date'].iloc[0].date()} -> "
          f"{df['date'].iloc[cut-1].date()}), holdout={len(X_hold)} rows "
          f"({df['date'].iloc[cut].date()} -> {df['date'].iloc[-1].date()})")

    model = xgb.XGBRegressor(**XGB_PARAMS, early_stopping_rounds=40)
    model.fit(X_train, y_train, eval_set=[(X_hold, y_hold)], verbose=False)
    pred_hold = model.predict(X_hold)
    evaluate(y_hold, pred_hold, "chronological holdout")

    # baseline for comparison: distance-only log-log regression
    base_coef = np.polyfit(X_train["log_distance"], y_train, 1)
    base_pred = np.polyval(base_coef, X_hold["log_distance"])
    evaluate(y_hold, base_pred, "baseline (distance-only)")

    # ---------- 2. time-series cross validation (robustness) ----------
    tscv = TimeSeriesSplit(n_splits=5)
    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), start=1):
        m = xgb.XGBRegressor(**{**XGB_PARAMS, "n_estimators": model.best_iteration or 300})
        m.fit(X.iloc[tr_idx], y[tr_idx])
        pred = m.predict(X.iloc[te_idx])
        metrics = evaluate(y[te_idx], pred, f"CV fold {fold}")
        fold_metrics.append(metrics)
    mape_scores = [m["mape_%"] for m in fold_metrics]
    print(f"CV MAPE mean={np.mean(mape_scores):.3f}%  std={np.std(mape_scores):.3f}%")

    # ---------- 3. final model on 100% of data ----------
    final_rounds = model.best_iteration + 1 if model.best_iteration else XGB_PARAMS["n_estimators"]
    final_params = {**XGB_PARAMS, "n_estimators": final_rounds}
    final_params.pop("early_stopping_rounds", None)
    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(X, y)

    MODELS.mkdir(exist_ok=True)
    final_model.save_model(MODELS / "rate_model.json")
    with open(MODELS / "market_proxy.pkl", "wb") as f:
        pickle.dump(market_proxy, f)
    with open(MODELS / "feature_cols.json", "w") as f:
        json.dump(ALL_FEATURE_COLS, f)
    with open(MODELS / "city_coords.json", "w") as f:
        json.dump(city_coords, f)

    # feature importance for the report / loom
    importances = pd.Series(final_model.feature_importances_, index=ALL_FEATURE_COLS).sort_values(ascending=False)
    print("\nFeature importances (final model):")
    print(importances.round(4))

    print(f"\nSaved model to {MODELS/'rate_model.json'}")
    print(f"Used {final_rounds} boosting rounds (selected via early stopping on the holdout).")


if __name__ == "__main__":
    main()
