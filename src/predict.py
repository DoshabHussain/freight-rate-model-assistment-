"""
Generate the two required prediction outputs:

  outputs/validation_predictions.csv      - load_id, predicted_rate  (12,000 rows)
  outputs/december_chart_inputs.csv       - the original 7 columns, predicted_rate filled (31 rows)

Usage:
    python -m src.predict
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from src.data_prep import build_features
except ImportError:
    from data_prep import build_features

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"


def load_artifacts():
    with open(MODELS / "feature_cols.json") as f:
        feature_cols = json.load(f)
    with open(MODELS / "market_proxy.pkl", "rb") as f:
        market_proxy = pickle.load(f)
    with open(MODELS / "city_coords.json") as f:
        city_coords = {k: tuple(v) for k, v in json.load(f).items()}
    model = xgb.XGBRegressor()
    model.load_model(MODELS / "rate_model.json")
    return model, market_proxy, city_coords, feature_cols


def predict_rate(df: pd.DataFrame, model, market_proxy, city_coords, feature_cols) -> np.ndarray:
    feats = build_features(df, market_proxy, city_coords)
    X = feats[feature_cols]
    log_pred = model.predict(X)
    return np.exp(log_pred)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    model, market_proxy, city_coords, feature_cols = load_artifacts()

    # ---------- validation.csv -> validation_predictions.csv ----------
    val = pd.read_csv(DATA / "validation.csv")
    val_preds = predict_rate(val, model, market_proxy, city_coords, feature_cols)

    template = pd.read_csv(DATA / "validation-predictions-template.csv")
    assert list(template["load_id"]) == list(val["load_id"]), "load_id order mismatch vs template"
    template["predicted_rate"] = np.round(val_preds, 2)
    template.to_csv(OUT / "validation_predictions.csv", index=False)
    print(f"Wrote {OUT/'validation_predictions.csv'} ({len(template)} rows)")

    # ---------- december-chart-inputs.csv -> filled ----------
    dec = pd.read_csv(DATA / "december-chart-inputs.csv")
    dec_preds = predict_rate(dec.drop(columns=["predicted_rate"]), model, market_proxy, city_coords, feature_cols)
    dec_out = dec.copy()
    dec_out["predicted_rate"] = np.round(dec_preds, 2)
    dec_out.to_csv(OUT / "december_chart_inputs.csv", index=False)
    print(f"Wrote {OUT/'december_chart_inputs.csv'} ({len(dec_out)} rows)")
    print(dec_out[["date", "predicted_rate"]].to_string(index=False))


if __name__ == "__main__":
    main()
