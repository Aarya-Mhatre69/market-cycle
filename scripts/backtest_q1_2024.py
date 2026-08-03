"""
Backtest the trained LightGBM price-band models on the first months of 2024.

Randomly selects 10 of the trained tickers (from the eval report's
per_ticker keys), engineers features from the full per-ticker history,
runs inference over the 2024-01-01 .. 2024-02-23 window (the data ends
there), writes a per-day results CSV and renders one prediction plot per
ticker.

Usage:
    $env:PYTHONPATH='src'; .venv\Scripts\python.exe scripts\backtest_q1_2024.py
"""

import logging
import random
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger("backtest_q1_2024")

from shankh.ml.company.config import CONFIG
from shankh.ml.company.features import add_features
from shankh.ml.company.inference import load_models_and_features, predict_bands
from shankh.ml.company.train import plot_predictions

_SEED = 42
_START = "2024-01-01"
_END = "2024-02-23"
_N_TICKERS = 10
_OUT_DIR = Path(CONFIG["artifacts"]["artifacts_dir"]) / "backtest_2024_q1"
_PLOT_DIR = _OUT_DIR / "plots"


def trained_tickers() -> list[str]:
    report_path = Path(CONFIG["artifacts"]["artifacts_dir"]) / CONFIG["artifacts"]["eval_report"]
    report = json_load(report_path)
    return sorted(report["per_ticker"].keys())


def json_load(path: Path) -> dict:
    import json

    with open(path, "r") as f:
        return json.load(f)


def load_single_ticker(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = df.columns.str.lower()
    df["ticker"] = csv_path.stem.upper()
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def run_ticker(
    ticker: str,
    upper_model,
    lower_model,
    feature_cols: list[str],
) -> pd.DataFrame:
    raw = load_single_ticker(Path(CONFIG["data"]["data_dir"]) / f"{ticker}.csv")
    feat = add_features(raw, cfg=CONFIG["features"])

    window = (feat["date"] >= _START) & (feat["date"] <= _END)
    test = feat[window].reset_index(drop=True)

    preds = predict_bands(test, upper_model, lower_model, feature_cols)

    actuals = test[["date", "ticker", "next_high_actual", "next_low_actual", "next_close_actual"]]
    merged = preds.merge(actuals, on=["date", "ticker"], how="left", suffixes=("", "_actual"))

    merged["close_in_band"] = (
        (merged["next_close_actual"] >= merged["pred_low_price"])
        & (merged["next_close_actual"] <= merged["pred_high_price"])
    ).astype(int)
    merged["high_within_band"] = (
        (merged["next_high_actual"] >= merged["pred_low_price"])
        & (merged["next_high_actual"] <= merged["pred_high_price"])
    ).astype(int)
    merged["low_within_band"] = (
        (merged["next_low_actual"] >= merged["pred_low_price"])
        & (merged["next_low_actual"] <= merged["pred_high_price"])
    ).astype(int)

    return merged


def main() -> None:
    rng = random.Random(_SEED)
    tickers = trained_tickers()
    selected = rng.sample(tickers, _N_TICKERS)

    logger.info("Selected 10 random trained tickers: %s", ", ".join(selected))

    upper_model, lower_model, feature_cols = load_models_and_features(CONFIG)

    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    frames = []
    for ticker in selected:
        logger.info("Backtesting %s over %s .. %s", ticker, _START, _END)
        res = run_ticker(ticker, upper_model, lower_model, feature_cols)
        frames.append(res)

        plot_path = _PLOT_DIR / f"{ticker}_prediction.png"
        plot_predictions(res, ticker=ticker, out_path=plot_path, show=False)
        logger.info("Saved plot -> %s", plot_path)

    all_results = pd.concat(frames, ignore_index=True).sort_values(["ticker", "date"])

    results_path = _OUT_DIR / "results_per_day.csv"
    all_results.to_csv(results_path, index=False)
    logger.info("Saved %d per-day result rows -> %s", len(all_results), results_path)

    summary_path = _OUT_DIR / "summary.csv"
    summary = (
        all_results.groupby("ticker")
        .agg(
            n_days=("date", "count"),
            close_coverage_pct=("close_in_band", lambda s: round(s.mean() * 100, 2)),
            high_within_pct=("high_within_band", lambda s: round(s.mean() * 100, 2)),
            low_within_pct=("low_within_band", lambda s: round(s.mean() * 100, 2)),
            mean_band_width_pct=("band_width_pct", "mean"),
            mean_close=("close", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)
    logger.info("Saved summary -> %s", summary_path)

    tickers_path = _OUT_DIR / "selected_tickers.json"
    import json

    with open(tickers_path, "w") as f:
        json.dump({"seed": _SEED, "selected_tickers": selected}, f, indent=2)
    logger.info("Saved ticker selection -> %s", tickers_path)


if __name__ == "__main__":
    sys.exit(main())
