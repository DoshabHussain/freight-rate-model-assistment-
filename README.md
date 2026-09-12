# Freight Rate Prediction

Predicts spot freight `posted_rate` from load features (lane, distance,
equipment, weight, date, and two market-condition signals) using XGBoost,
plus a small auxiliary model for extrapolating market conditions into
Nov/Dec dates that fall outside the training window.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # optional but recommended
pip install -r requirements.txt
```

## Run

```bash
# 1. Explore the data (prints the key findings referenced in the report)
python -m src.eda

# 2. Train the model (chronological holdout + 5-fold TimeSeriesSplit CV,
#    then refits on 100% of train-test.csv). Saves artifacts to models/.
python -m src.train

# 3. Generate predictions for both required output files
python -m src.predict
#   -> outputs/validation_predictions.csv       (load_id, predicted_rate)
#   -> outputs/december_chart_inputs.csv        (original 7 cols, predicted_rate filled)

# 4. Run the provided scorer / chart generator
python score.py \
  --predictions outputs/validation_predictions.csv \
  --december-predictions outputs/december_chart_inputs.csv \
  --output-dir scorer_results
#   -> scorer_results/candidate_december.png
```

## Repository layout

```
data/                       provided input CSVs (unchanged)
src/
  data_prep.py              cleaning, feature engineering, shared by train & predict
  market_proxy.py           harmonic regression used to extrapolate market_index/
                             quote_signal to dates outside the training range
  train.py                  training + validation + saves model artifacts
  predict.py                loads artifacts, produces both output CSVs
  eda.py                    reproducible exploration / data-quality findings
models/                     saved model + auxiliary artifacts (created by train.py)
outputs/                    generated prediction CSVs (created by predict.py)
score.py                    provided scorer (unmodified)
requirements.txt
```

## Approach summary

**Target:** `log(posted_rate)`, modeled with gradient-boosted trees
(XGBoost, native categorical support for `equipment`).

**Validation:** train/hold-out split is done *chronologically* (train on
the first ~85% of dates, hold out the most recent ~15%) rather than a
random split, because the actual task is forecasting into Nov/Dec — dates
strictly after the training window. A random split would leak nearby-day
market conditions into training and overstate accuracy. A 5-fold expanding
`TimeSeriesSplit` cross-validation confirms the holdout number isn't a
fluke of where the cut falls. The final shipped model is refit on 100% of
`train-test.csv` once the approach was validated this way.

**Key data-quality issues found and fixes (see `src/eda.py` for the
reproducible checks):**
1. **`weight` sign corruption** — ~0.6% of rows have negative weight, but
   the distribution of `abs(weight)` for those rows is statistically
   identical to the normal positive rows (same mean/range) → sign-flip
   entry error, fixed by taking the absolute value (with a missing-flag
   column for genuinely missing weights).
2. **Missing `market_index` / `quote_signal`** — imputed using the mean of
   other loads on the *same date* first (these are mostly date-level
   signals with small load-to-load noise), falling back to a fitted trend
   model only when no same-date value exists at all.
3. **`december-chart-inputs.csv` has no `market_index`/`quote_signal`
   columns whatsoever**, and its dates (Dec 2025) are entirely outside the
   training window (Jan–Oct 2025). Tree ensembles don't extrapolate
   trends — a raw month/day feature for December would just get bucketed
   with whatever the nearest training month looked like. Instead, a small
   harmonic (Fourier) regression is fit on the *daily average* market
   signals from the training data and used to extrapolate a smooth,
   periodic value for any future date. This keeps the main model's date
   handling limited to safely-recurring features (day-of-week), while the
   proxy model owns the actual out-of-range extrapolation.
4. **New pickup/delivery cities appear in `validation.csv` that never
   appear in `train-test.csv`** (Charlotte, Chicago, Jackson, Laredo,
   Knoxville, San Diego, Allentown, Norfolk). Using city name as a
   categorical feature would hit unseen categories at inference time, so
   the model uses continuous pickup/delivery latitude & longitude instead,
   which generalizes to any new city. (`december-chart-inputs.csv` doesn't
   even provide lat/lon — only city names — so a lookup table built from
   `train-test.csv`'s city→coordinate pairs fills those in for the two
   cities it uses, Lexington and Fort Wayne, both seen in training.)

**Model performance (chronological holdout, last ~7,200 rows / Sep 15 –
Oct 31):** MAE ≈ $141, MAPE ≈ 6.1%, R² ≈ 0.84, versus a distance-only
log-log baseline of MAE ≈ $186 / MAPE ≈ 8.2% / R² ≈ 0.83. `log(distance)`
is by far the dominant driver (R²=0.94 on its own); lane geography
(lat/lon), equipment type, weight, and the market signals provide the
remaining lift. `quote_signal` in particular carries very little
information once distance is accounted for (correlation with the
distance-adjusted residual ≈ 0.04) and is kept mainly as low-weight signal
rather than a primary driver.
