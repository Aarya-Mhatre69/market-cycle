# price_band — Module Guide for New Engineers

This module builds a **production-grade quantile price-band forecasting pipeline** for equities.
Given today's OHLCV bar, it predicts tomorrow's likely high/low price range as a calibrated
statistical interval — not a single point forecast.

---

## What it does in one sentence

> For each trading day, predict a `[pred_low_price, pred_high_price]` band such that
> tomorrow's close lands inside it with a target probability (default ~68 %).

---

## Key design decisions (read these first)

| Decision | Why |
|---|---|
| **Quantile regression, not MSE** | Pinball loss directly optimises interval coverage. MSE would give point forecasts and say nothing about uncertainty. |
| **Instrument-agnostic model** | `ticker` is used only to group rows during feature computation. It is never a model input. The same trained model works on any equity in any market. |
| **Log-return targets** | `target_upper = log(next_high / close)`, `target_lower = log(next_low / close)`. Log-returns are stationary and scale-invariant across price levels. |
| **Walk-forward (expanding window) CV** | Financial time series cannot be shuffled. Each fold trains on all data up to a cutoff date and validates on the next window — no future data ever leaks into training. |
| **Optuna HPO per target** | Upper and lower band models are tuned independently because their distributions are asymmetric (high > close always; low < close always). |
| **Post-processing guarantee** | `pred_low ≤ pred_high` is enforced after every prediction. The model can occasionally predict them inverted — we sort and apply a minimum width floor. |

---

## Directory layout

```
src/shankh/price_band/
│
├── README.md               ← you are here
├── __init__.py             ← public surface (import shortcuts)
│
├── config.py               ← ALL configuration lives here
├── data_loader.py          ← read CSV files → validated DataFrame
├── features.py             ← OHLCV → 107 engineered features + targets
├── validation.py           ← walk-forward CV fold generation
├── evaluation.py           ← interval metrics, baselines, report saving
├── inference.py            ← predict bands from trained models
├── pipeline.py             ← top-level orchestrator (run everything)
│
└── models/
    ├── model.py            ← LightGBM / XGBoost quantile model builders
    └── trainer.py          ← Optuna HPO, final training, artifact saving
```

---

## File-by-file reference

### `config.py`
Single source of truth for every tunable knob.

```python
from shankh.price_band.config import CONFIG

CONFIG["model"]["backend"]          # "lightgbm" or "xgboost"
CONFIG["model"]["upper_quantile"]   # 0.84  → upper band target
CONFIG["model"]["lower_quantile"]   # 0.16  → lower band target
CONFIG["data"]["test_cutoff"]       # "2024-07-01" — hard OOS test start
CONFIG["walk_forward"]["n_splits"]  # 5 CV folds
CONFIG["optuna"]["n_trials"]        # 40 HPO trials per target
```

**Never hardcode paths or hyperparameters elsewhere** — always edit here and import `CONFIG`.

---

### `data_loader.py`
Reads all `*.csv` files from `data/`, validates schema and OHLC ordering constraints,
removes duplicate `(ticker, date)` pairs, and returns a single combined DataFrame.

```python
from shankh.price_band.data_loader import load_ohlcv

df = load_ohlcv("path/to/data/")
# columns: date, open, high, low, close, volume, ticker
```

Expected CSV format per file (one file per ticker, filename = ticker):
```
date,open,high,low,close,volume
2015-01-01,3511.28,3586.77,3464.27,3515.23,4122319
```

---

### `features.py`
Takes the raw OHLCV DataFrame and adds ~100 engineered features.
`ticker` is used only as a groupby key — it never appears in the model feature set.

**Feature groups:**

| Group | Examples |
|---|---|
| Returns / momentum | `ret_1`, `ret_5`, `cum_ret_20`, `gap_open` |
| Candlestick anatomy | `hl_range`, `oc_body`, `upper_wick`, `lower_wick`, `body_to_span` |
| Volume / liquidity | `log_volume`, `vol_ratio_20`, `dv_zscore_20` |
| Trend / MA | `ma_5/10/20/50`, `ema_5/10/20/50`, `zscore_20`, `ma_cross_5_20` |
| Oscillators | `rsi_14`, `stoch_k_14`, `williams_r_14`, `macd_norm`, `macd_hist` |
| Volatility | `atr_pct`, `rv_5/10/20`, `parkinson_20`, `garman_klass_20`, `bb_pct_b`, `bb_width` |
| Breakout | `breakout_up_10`, `breakdown_dn_20`, `range_pct_50` |
| Calendar | `dayofweek`, `month`, `is_month_end`, `is_quarter_end` |
| Regime proxies | `vov_20`, `ret_skew_20`, `ret_autocorr_20`, `drawdown_50` |

**Targets** (appended last, excluded from model inputs):
- `target_upper` = `log(next_high / close)`
- `target_lower` = `log(next_low / close)`
- `next_high_actual`, `next_low_actual`, `next_close_actual` — used only in evaluation

```python
from shankh.price_band.features import add_features, get_feature_cols

feat_df      = add_features(raw_df, cfg=CONFIG["features"])
feature_cols = get_feature_cols(feat_df)   # list of ~107 column names
```

---

### `validation.py`
Generates expanding-window walk-forward CV folds.

```
Timeline example (5 folds, val_size=60 dates, gap=1):

Fold 0:  [=====TRAIN=====] [gap] [==VAL==]
Fold 1:  [=========TRAIN=========] [gap] [==VAL==]
Fold 2:  [==============TRAIN==============] [gap] [==VAL==]
...
```

- `gap=1` means one trading date is skipped between train and val — prevents any
  rolling feature from looking ahead across the fold boundary.
- Folds work on **unique dates**, not rows, so multi-ticker frames split consistently.
- `train_idx` and `val_idx` are **positional iloc indices** into the DataFrame you passed in.

```python
from shankh.price_band.validation import walk_forward_folds

folds = walk_forward_folds(pretrain_df, n_splits=5, val_size=60, min_train_size=250, gap=1)
# folds[0].train_idx  → numpy array of row positions
# folds[0].val_start  → pd.Timestamp
```

---

### `models/model.py`
Builds quantile regression models. Supports LightGBM (`objective="quantile"`, pinball loss)
and XGBoost (`objective="reg:quantileerror"`).

```python
from shankh.price_band.models.model import build_model

upper_model = build_model("lightgbm", params, quantile=0.84, seed=42)
lower_model = build_model("lightgbm", params, quantile=0.16, seed=42)
```

**Why two separate models?**
The upper band targets the 84th percentile of `log(next_high/close)` — almost always positive.
The lower band targets the 16th percentile of `log(next_low/close)` — almost always negative.
Their feature importances and optimal hyperparameters differ, so they are tuned independently.

---

### `models/trainer.py`
Three responsibilities:

**1. Optuna HPO (`run_hpo`)**
Runs `n_trials` Optuna trials. Each trial trains one model per CV fold and scores it with
pinball loss. The best params minimise mean pinball loss across all folds.

**2. Final training (`train_final_models`)**
Trains the final model on 90% of pre-test data. The last 10% is used only for early stopping
(not for hyperparameter selection — no leakage).

**3. CV diagnostics (`collect_cv_metrics`)**
Re-trains with best params on each fold and records: pinball loss, MAE/RMSE, coverage %,
band width. This is your main tool for diagnosing overfitting or regime sensitivity.

```python
study, best_params = run_hpo(pretrain_df, feature_cols, "target_upper", 0.84, folds, cfg)
upper_model, lower_model = train_final_models(pretrain_df, feature_cols, best_upper, best_lower, cfg)
cv_results = collect_cv_metrics(pretrain_df, feature_cols, best_upper, best_lower, folds, cfg)
```

---

### `evaluation.py`
Computes a full suite of interval-forecasting metrics on the held-out test set,
plus two baselines for comparison.

**Metrics:**

| Metric | What it measures |
|---|---|
| `close_coverage_pct` | % of days where next-day close lands inside the band |
| `envelope_coverage_pct` | % of days where the full [low, high] bar fits inside the band |
| `calibration_error` | \|empirical coverage − nominal coverage\| — should be near 0 |
| `mean_band_width_pct` | Average band width as % of close — sharpness measure |
| `mean_winkler_score` | Width + penalty for misses — composite interval score (lower = better) |
| `pinball_upper/lower` | Quantile loss for each band — the training objective |
| `mae/rmse_high/low_price` | Price-space accuracy vs actual next-day high/low |

**Baselines:**
- **Persistence**: predict tomorrow's band = today's high/low
- **Constant-vol**: symmetric band using 20-day realised volatility scaled by a normal z-score

The model must beat both on Winkler score and coverage to be considered useful.

---

### `inference.py`
Loads trained models and runs predictions. Enforces `pred_low ≤ pred_high` via sorting
and applies a minimum band width floor (0.1% of close).

```python
from shankh.price_band.inference import predict_bands, load_models_and_features

upper_model, lower_model, feature_cols = load_models_and_features(cfg)
preds = predict_bands(feat_df, upper_model, lower_model, feature_cols)
# preds columns: date, ticker, close, pred_upper_ret, pred_lower_ret,
#                pred_high_price, pred_low_price, band_width_pct
```

---

### `pipeline.py`
Runs all 11 steps end-to-end. This is the main entry point.

```bash
# from src/
uv run python -m shankh.price_band.pipeline

# with config overrides (JSON file merged onto defaults)
uv run python -m shankh.price_band.pipeline --config overrides.json
```

**Steps:**
```
1.  Load OHLCV CSVs
2.  Feature engineering (107 features)
3.  Train/test split at test_cutoff
4.  Walk-forward CV folds from pre-test data
5.  Optuna HPO → upper band
6.  Optuna HPO → lower band
7.  CV diagnostics with best params
8.  Final model training on full pre-test data
9.  Inference on held-out test set
10. Full evaluation + baseline comparison
11. Save all artifacts
```

---

## Artifacts produced

All saved to `models/price_band/artifacts/`:

| File | Contents |
|---|---|
| `upper_model.txt` / `lower_model.txt` | Trained LightGBM models (native format) |
| `feature_cols.pkl` | Ordered list of feature column names |
| `optuna_best.json` | Best hyperparameters + pinball score per target |
| `optuna_study_upper.pkl` / `_lower.pkl` | Full Optuna study objects (resumable) |
| `cv_results.json` | Per-fold metrics table |
| `eval_report.json` | Full test evaluation: overall + per-ticker + baselines |
| `predictions.parquet` | Test set predictions with actuals side-by-side |
| `feature_importance_upper.csv` / `_lower.csv` | Gain-based feature importance |

---

## Typical results (mock data, 5 tickers, 2015–2025)

```
close_coverage_pct    96.5%   (nominal target: 68%)   ← bands are conservative
mean_band_width_pct    8.5%
pinball_upper         0.0062
pinball_lower         0.0068
mae_high_price        71.4    vs persistence 61.8      ← expected: mock data has no real signal
mean_winkler_score   271.9    vs persistence 232.5
```

The model is conservative (over-covers) because quantiles 0.84/0.16 produce a wide band.
Tighten to 0.75/0.25 for sharper bands closer to the actual daily range.

---

## How to add a new feature

1. Open `features.py`
2. Add a new `_my_feature_group(df, g) -> dict[str, pd.Series]` function
3. Call it in `add_features()` and merge the result into `new_cols`
4. Re-run the pipeline — `get_feature_cols()` picks it up automatically

**Rules:**
- Compute within-ticker only (always use `g = df.groupby("ticker", ...)`)
- No look-ahead (never use `shift(-n)` for `n > 0` on features — only targets use it)
- Return a `dict` of `{column_name: pd.Series}`, never assign directly to `df`

---

## How to change the backend (XGBoost ↔ LightGBM)

Edit one line in `config.py`:
```python
"backend": "xgboost",   # or "lightgbm"
```

The search space used for HPO switches automatically (`xgb_space` vs `lgb_space`).

---

## Common gotchas

| Symptom | Likely cause |
|---|---|
| `No CSV files found` | Run from `src/` directory, or fix `data_dir` in `config.py` |
| `Not enough dates for walk-forward CV` | Reduce `min_train_size` or `val_size` in config |
| Feature name warnings from LightGBM | Passing `.values` (numpy) instead of a DataFrame — always pass DataFrames to `.fit()` and `.predict()` |
| `pred_high < pred_low` | Should not happen — `inference.py` sorts them — but if you see it in raw model output, it's normal for quantile regressors to cross occasionally |
| Coverage ≫ nominal | Quantile targets are wide (0.84/0.16); tighten toward 0.75/0.25 for sharper bands |
