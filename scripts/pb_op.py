"""
Daily price-band rendering pipeline.

Parses eval_report.json to extract active tickers, loads raw OHLCV data with
a historical warmup buffer, engineers features, executes LightGBM quantile band predictions
over the full test set (at least 30+ trading days), exports full test predictions to CSV,
and renders 2 identical 15-day daily price band plots (Days 1-15 and Days 16-30) for 5 selected tickers.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from shankh.agents.equity.pb_config import CONFIG
from shankh.agents.equity.pb_data_loader import load_ohlcv
from shankh.agents.equity.pb_features import add_features
from shankh.agents.equity.pb_inference import load_models_and_features, predict_bands

logger = logging.getLogger(__name__)


def extract_tickers_from_eval_json(eval_json_path: Path) -> list[str]:
    """
    Parse the evaluation report JSON and extract ticker symbols from 'per_ticker'.

    Parameters
    ----------
    eval_json_path : Path
        Path to the eval_report.json file.

    Returns
    -------
    list[str]
        List of ticker symbol strings found in the report.
    """
    if not eval_json_path.exists():
        raise FileNotFoundError(f"Evaluation report not found at: {eval_json_path}")

    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "per_ticker" not in data or not data["per_ticker"]:
        raise KeyError(f"No 'per_ticker' key found in evaluation JSON: {eval_json_path}")

    tickers = list(data["per_ticker"].keys())
    logger.info("Extracted %d tickers from %s", len(tickers), eval_json_path.name)
    return tickers


def _render_15d_price_band_plot(
    plot_data: pd.DataFrame,
    ticker: str,
    window_label: str,
    output_path: Path,
) -> None:
    """
    Render a single 15-day price band plot where every single trading day is
    explicitly rendered on the X-axis without binning or date grouping.

    Parameters
    ----------
    plot_data : pd.DataFrame
        15-row slice of daily predictions for a single ticker.
    ticker : str
        Ticker symbol.
    window_label : str
        Label indicating window range (e.g., 'Days 1-15' or 'Days 16-30').
    output_path : Path
        Target PNG file path.
    """
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)

    # Discrete x-axis positions (0 to 14) to prevent Matplotlib date binning/grouping
    n_days = len(plot_data)
    x_positions = np.arange(n_days)
    
    # Format date labels as explicit 'DD MMM' strings (e.g., '03 Feb')
    date_labels = [pd.to_datetime(d).strftime("%d %b") for d in plot_data["date"]]

    close_prices = plot_data["close"].values
    pred_high = plot_data["pred_high_price"].values
    pred_low = plot_data["pred_low_price"].values

    # 1. Shaded Interval Area
    ax.fill_between(
        x_positions,
        pred_low,
        pred_high,
        color="#1f77b4",
        alpha=0.22,
        label="Predicted Price Band Interval",
    )

    # 2. Predicted Upper Band Line
    ax.plot(
        x_positions,
        pred_high,
        color="#d62728",
        linestyle="--",
        linewidth=1.8,
        marker="^",
        markersize=5,
        label="Predicted Upper Band (High)",
    )

    # 3. Predicted Lower Band Line
    ax.plot(
        x_positions,
        pred_low,
        color="#2ca02c",
        linestyle="--",
        linewidth=1.8,
        marker="v",
        markersize=5,
        label="Predicted Lower Band (Low)",
    )

    # 4. Actual Spot Close Line & Markers
    ax.plot(
        x_positions,
        close_prices,
        color="#1f77b4",
        linestyle="-",
        linewidth=2.2,
        marker="o",
        markersize=7,
        label="Actual Spot Close",
    )

    # 5. Callout Annotation for the Final Day in the Window
    last_idx = n_days - 1
    ax.annotate(
        f"₹{close_prices[last_idx]:.2f}\n[{pred_low[last_idx]:.2f} - {pred_high[last_idx]:.2f}]",
        xy=(last_idx, close_prices[last_idx]),
        xytext=(-45, 15),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.7),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", color="black"),
    )

    # Title & Formatting
    start_str = pd.to_datetime(plot_data['date'].iloc[0]).strftime('%d %b %Y')
    end_str = pd.to_datetime(plot_data['date'].iloc[-1]).strftime('%d %b %Y')
    
    ax.set_title(
        f"{ticker} — Daily Price Band Forecast ({window_label})\n"
        f"Trading Dates: {start_str} to {end_str}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Trading Date (Individual Days)", fontsize=11, labelpad=8)
    ax.set_ylabel("Price (INR)", fontsize=11, labelpad=8)

    # Explicit discrete X-ticks for EVERY single trading day (zero binning/grouping)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(date_labels, rotation=45, ha="right", fontsize=9.5)

    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()

    fig.savefig(output_path, dpi=300)
    plt.close(fig)  # Isolated canvas release
    logger.info("Saved 15-day plot (%s) to %s", window_label, output_path)


def render_daily_forecasts_and_plots(
    eval_json_path: Optional[Path] = None,
    num_selected_tickers: int = 10,
    num_plot_tickers: int = 5,
    min_test_trading_days: int = 30,
) -> tuple[pd.DataFrame, list[Path]]:
    """
    Execute full rendering pipeline: load data with historical warmup, build features,
    predict daily price bands over full test set, export CSV, and render 2 identical
    15-day price band plots (Days 1-15 and Days 16-30) for 5 selected tickers.

    Parameters
    ----------
    eval_json_path : Path, optional
        Path to eval_report.json. Defaults to ARTIFACTS_DIR/eval_report.json.
    num_selected_tickers : int
        Number of tickers to select from report (default: 10).
    num_plot_tickers : int
        Number of tickers to render visual plots for (default: 5).
    min_test_trading_days : int
        Minimum trading days required in the test set evaluation slice (default: 30).

    Returns
    -------
    tuple[pd.DataFrame, list[Path]]
        Full test predictions DataFrame and list of generated plot file paths.
    """
    artifacts_dir = Path(CONFIG["artifacts"]["artifacts_dir"])
    if eval_json_path is None:
        eval_json_path = artifacts_dir / CONFIG["artifacts"]["eval_report"]

    # 1. Extract tickers from evaluation report
    all_tickers = extract_tickers_from_eval_json(eval_json_path)
    selected_tickers = all_tickers[:num_selected_tickers]
    logger.info("Selected %d tickers for pipeline: %s", len(selected_tickers), selected_tickers)

    # 2. Load raw OHLCV data (includes historical warmup data for rolling windows)
    raw_df = load_ohlcv(
        data_dir=CONFIG["data"]["data_dir"],
        min_tickers=len(selected_tickers),
        max_tickers=len(selected_tickers),
    )
    
    # Filter raw data to selected tickers only
    raw_df = raw_df[raw_df["ticker"].isin(selected_tickers)].copy()
    raw_df = raw_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 3. Compute Features across full historical dataset (warmup buffer consumes rolling NaNs)
    logger.info("Engineering features for %d tickers (with warmup buffer)...", len(selected_tickers))
    feat_df = add_features(raw_df, cfg=CONFIG["features"])

    # 4. Load Trained Models and Feature Columns
    upper_model, lower_model, feature_cols = load_models_and_features(CONFIG)

    # 5. Execute Daily Inference across all feature-engineered rows
    logger.info("Running LightGBM daily price-band prediction...")
    preds_df = predict_bands(
        df=feat_df,
        upper_model=upper_model,
        lower_model=lower_model,
        feature_cols=feature_cols,
    )

    # Attach actual OHLC details for evaluation and plot context
    preds_df["open"] = feat_df["open"].values
    preds_df["high"] = feat_df["high"].values
    preds_df["low"] = feat_df["low"].values

    # 6. Filter to Full Test Dataset Scope (at least 30+ trading days)
    test_cutoff = pd.Timestamp(CONFIG["data"]["test_cutoff"])
    test_preds = preds_df[preds_df["date"] >= test_cutoff].copy()

    # Enforce minimum test set duration constraint (at least min_test_trading_days)
    unique_test_dates = test_preds["date"].nunique()
    if unique_test_dates < min_test_trading_days:
        logger.warning(
            "Test split from cutoff %s has only %d trading days (< %d). Expanding to last %d trading days.",
            test_cutoff.date(), unique_test_dates, min_test_trading_days, min_test_trading_days
        )
        all_unique_dates = sorted(preds_df["date"].unique())
        effective_start_date = all_unique_dates[-min_test_trading_days]
        test_preds = preds_df[preds_df["date"] >= effective_start_date].copy()

    test_preds = test_preds.sort_values(["ticker", "date"]).reset_index(drop=True)
    logger.info(
        "Full test dataset scope: %d total prediction rows across %d unique trading dates (%s to %s).",
        len(test_preds),
        test_preds["date"].nunique(),
        test_preds["date"].min().date(),
        test_preds["date"].max().date(),
    )

    # Output directory for renders and CSV artifact
    render_dir = artifacts_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    # 7. Save Full Test Predictions as CSV File
    csv_path = render_dir / "full_test_predictions.csv"
    export_cols = [
        "date", "ticker", "open", "high", "low", "close",
        "pred_high_price", "pred_low_price", "band_width_pct",
        "pred_upper_ret", "pred_lower_ret", "band_ordering_ok"
    ]
    available_export_cols = [c for c in export_cols if c in test_preds.columns]
    test_preds[available_export_cols].to_csv(csv_path, index=False)
    logger.info("Saved FULL test dataset predictions (%d rows) to CSV: %s", len(test_preds), csv_path)

    # 8. Render 2 Identical 15-Day Price Band Plots (Days 1-15 & Days 16-30) for 5 Tickers
    plot_tickers = selected_tickers[:num_plot_tickers]
    generated_plots: list[Path] = []

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    for ticker in plot_tickers:
        tkr_test_data = test_preds[test_preds["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        total_days = len(tkr_test_data)

        # Slice 30 days into two contiguous 15-day chunks
        if total_days >= 30:
            window1_data = tkr_test_data.iloc[-30:-15].reset_index(drop=True)
            window2_data = tkr_test_data.iloc[-15:].reset_index(drop=True)
        else:
            mid = total_days // 2
            window1_data = tkr_test_data.iloc[:mid].reset_index(drop=True)
            window2_data = tkr_test_data.iloc[mid:].reset_index(drop=True)

        # Plot 1: Days 1 to 15 (First Half of Month)
        plot1_path = render_dir / f"{ticker}_plot1_days01_to_15.png"
        _render_15d_price_band_plot(
            plot_data=window1_data,
            ticker=ticker,
            window_label="Days 1 to 15",
            output_path=plot1_path,
        )
        generated_plots.append(plot1_path)

        # Plot 2: Days 16 to 30 (Second Half of Month)
        plot2_path = render_dir / f"{ticker}_plot2_days16_to_30.png"
        _render_15d_price_band_plot(
            plot_data=window2_data,
            ticker=ticker,
            window_label="Days 16 to 30",
            output_path=plot2_path,
        )
        generated_plots.append(plot2_path)

    return test_preds, generated_plots


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    render_daily_forecasts_and_plots()