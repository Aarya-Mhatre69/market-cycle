"""
Evaluation module for the price-band forecasting pipeline.

Computes a comprehensive set of interval-forecasting metrics:

Calibration
-----------
- Empirical coverage at the nominal level (close_coverage, envelope_coverage)
- Coverage calibration error: |empirical_coverage − nominal_coverage|

Sharpness
---------
- Mean and percentile band width in price space and as % of close

Accuracy (return-space)
-----------------------
- Pinball loss (upper and lower)
- MAE, RMSE in log-return space

Accuracy (price-space)
----------------------
- MAE, RMSE of predicted high/low vs actual next-day high/low

Baseline comparison
-------------------
- Persistence baseline: predict today's high/low as tomorrow's band
- Constant-volatility baseline: use realised vol to build a symmetric band

All metrics are computed over the full test set and also broken down
by ticker for diagnostics.
"""


import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate(
    test_df: pd.DataFrame,
    pred_upper_ret: np.ndarray,
    pred_lower_ret: np.ndarray,
    cfg: dict,
) -> dict:
    """
    Full evaluation of interval forecasts on the held-out test set.

    Parameters
    ----------
    test_df : pd.DataFrame
        Test rows with columns: close, next_high_actual, next_low_actual,
        next_close_actual, target_upper, target_lower, ticker, date.
    pred_upper_ret : np.ndarray
        Predicted upper log-return (log(pred_high / close)).
    pred_lower_ret : np.ndarray
        Predicted lower log-return (log(pred_low / close)).
    cfg : dict
        Full CONFIG dict.

    Returns
    -------
    dict
        Nested dict: "overall", "per_ticker", "baseline_persistence",
        "baseline_constant_vol".
    """
    upper_q = cfg["model"]["upper_quantile"]
    lower_q = cfg["model"]["lower_quantile"]
    nominal_coverage = cfg["evaluation"]["target_coverage"]
    sharpness_q      = cfg["evaluation"]["sharpness_quantile"]

    # Enforce ordering: band_upper ≥ band_lower  (post-processing)
    band_upper = np.maximum(pred_upper_ret, pred_lower_ret)
    band_lower = np.minimum(pred_upper_ret, pred_lower_ret)

    close  = test_df["close"].values
    act_h  = test_df["next_high_actual"].values
    act_l  = test_df["next_low_actual"].values
    act_c  = test_df["next_close_actual"].values
    tgt_u  = test_df["target_upper"].values
    tgt_l  = test_df["target_lower"].values

    pred_high = close * np.exp(band_upper)
    pred_low  = close * np.exp(band_lower)

    overall = _compute_metrics(
        tgt_u, tgt_l,
        pred_upper_ret, pred_lower_ret,
        band_upper, band_lower,
        pred_high, pred_low,
        act_h, act_l, act_c, close,
        upper_q, lower_q, nominal_coverage, sharpness_q,
    )

    # Per-ticker breakdown
    per_ticker: dict[str, dict] = {}
    for tkr, grp in test_df.groupby("ticker"):
        idx = grp.index
        per_ticker[str(tkr)] = _compute_metrics(
            tgt_u[idx], tgt_l[idx],
            pred_upper_ret[idx], pred_lower_ret[idx],
            band_upper[idx], band_lower[idx],
            pred_high[idx], pred_low[idx],
            act_h[idx], act_l[idx], act_c[idx], close[idx],
            upper_q, lower_q, nominal_coverage, sharpness_q,
        )

    # Baselines
    baseline_persistence  = _baseline_persistence(test_df, nominal_coverage, sharpness_q)
    baseline_constant_vol = _baseline_constant_vol(test_df, nominal_coverage, sharpness_q)

    report = {
        "overall":              overall,
        "per_ticker":           per_ticker,
        "baseline_persistence": baseline_persistence,
        "baseline_constant_vol": baseline_constant_vol,
    }
    logger.info(
        "Test evaluation | close_coverage=%.1f%% (nominal=%.0f%%) | "
        "mean_band_width=%.2f%% | pinball_upper=%.5f | pinball_lower=%.5f",
        overall["close_coverage_pct"],
        nominal_coverage * 100.0,
        overall["mean_band_width_pct"],
        overall["pinball_upper"],
        overall["pinball_lower"],
    )
    return report


# ---------------------------------------------------------------------------
# Core metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(
    tgt_u, tgt_l,
    pred_u_ret, pred_l_ret,
    band_u, band_l,
    pred_high, pred_low,
    act_h, act_l, act_c, close,
    upper_q, lower_q,
    nominal_coverage: float,
    sharpness_q: float,
) -> dict:
    close_inside    = (act_c >= pred_low) & (act_c <= pred_high)
    envelope_inside = (act_l >= pred_low) & (act_h <= pred_high)
    band_width_pct  = (pred_high - pred_low) / (close + 1e-12) * 100.0

    empirical_coverage = float(close_inside.mean())
    calibration_error  = abs(empirical_coverage - nominal_coverage)

    # Winkler score: width + 2/α penalty for misses (commonly used for intervals)
    alpha = 1.0 - nominal_coverage
    winkler = _winkler_score(pred_high, pred_low, act_c, alpha)

    return {
        # Calibration
        "close_coverage_pct":      float(close_inside.mean() * 100.0),
        "envelope_coverage_pct":   float(envelope_inside.mean() * 100.0),
        "nominal_coverage_pct":    float(nominal_coverage * 100.0),
        "calibration_error":       float(calibration_error),
        # Sharpness
        "mean_band_width_pct":     float(band_width_pct.mean()),
        "median_band_width_pct":   float(np.median(band_width_pct)),
        f"p{int(sharpness_q*100)}_band_width_pct": float(np.percentile(band_width_pct, sharpness_q * 100)),
        # Accuracy – return space
        "pinball_upper":    _pinball(tgt_u, pred_u_ret, upper_q),
        "pinball_lower":    _pinball(tgt_l, pred_l_ret, lower_q),
        "mae_upper_ret":    float(mean_absolute_error(tgt_u, pred_u_ret)),
        "mae_lower_ret":    float(mean_absolute_error(tgt_l, pred_l_ret)),
        "rmse_upper_ret":   float(np.sqrt(mean_squared_error(tgt_u, pred_u_ret))),
        "rmse_lower_ret":   float(np.sqrt(mean_squared_error(tgt_l, pred_l_ret))),
        # Accuracy – price space
        "mae_high_price":   float(mean_absolute_error(act_h, pred_high)),
        "mae_low_price":    float(mean_absolute_error(act_l, pred_low)),
        "rmse_high_price":  float(np.sqrt(mean_squared_error(act_h, pred_high))),
        "rmse_low_price":   float(np.sqrt(mean_squared_error(act_l, pred_low))),
        # Composite interval score
        "mean_winkler_score": float(winkler),
        "n_samples":          int(len(act_c)),
    }


# ---------------------------------------------------------------------------
# Baseline: persistence
# ---------------------------------------------------------------------------

def _baseline_persistence(
    test_df: pd.DataFrame,
    nominal_coverage: float,
    sharpness_q: float,
) -> dict:
    """
    Predict tomorrow's high/low = today's high/low.
    A trivially simple benchmark that any model must beat.
    """
    close  = test_df["close"].values
    act_h  = test_df["next_high_actual"].values
    act_l  = test_df["next_low_actual"].values
    act_c  = test_df["next_close_actual"].values

    pred_high = test_df["high"].values   # today's high as prediction
    pred_low  = test_df["low"].values    # today's low  as prediction

    close_inside    = (act_c >= pred_low) & (act_c <= pred_high)
    envelope_inside = (act_l >= pred_low) & (act_h <= pred_high)
    band_width_pct  = (pred_high - pred_low) / (close + 1e-12) * 100.0

    alpha   = 1.0 - nominal_coverage
    winkler = _winkler_score(pred_high, pred_low, act_c, alpha)

    return {
        "close_coverage_pct":    float(close_inside.mean() * 100.0),
        "envelope_coverage_pct": float(envelope_inside.mean() * 100.0),
        "mean_band_width_pct":   float(band_width_pct.mean()),
        f"p{int(sharpness_q*100)}_band_width_pct": float(np.percentile(band_width_pct, sharpness_q * 100)),
        "mae_high_price":        float(mean_absolute_error(act_h, pred_high)),
        "mae_low_price":         float(mean_absolute_error(act_l, pred_low)),
        "mean_winkler_score":    float(winkler),
    }


# ---------------------------------------------------------------------------
# Baseline: constant-volatility symmetric band
# ---------------------------------------------------------------------------

def _baseline_constant_vol(
    test_df: pd.DataFrame,
    nominal_coverage: float,
    sharpness_q: float,
) -> dict:
    """
    Symmetric band: ±z * σ where σ is the 20-day realised vol
    and z is chosen to match the nominal coverage level under normality.
    """
    from scipy.stats import norm  # local import — optional dep

    z = norm.ppf((1.0 + nominal_coverage) / 2.0)   # e.g. ≈1.0 for 68%

    # rv_20 should already be in the frame from feature engineering
    if "rv_20" not in test_df.columns:
        logger.warning("rv_20 not in test_df — constant_vol baseline unavailable.")
        return {}

    close    = test_df["close"].values
    sigma    = test_df["rv_20"].values / np.sqrt(252)   # daily vol
    act_h    = test_df["next_high_actual"].values
    act_l    = test_df["next_low_actual"].values
    act_c    = test_df["next_close_actual"].values

    pred_high = close * np.exp(z * sigma)
    pred_low  = close * np.exp(-z * sigma)

    close_inside    = (act_c >= pred_low) & (act_c <= pred_high)
    envelope_inside = (act_l >= pred_low) & (act_h <= pred_high)
    band_width_pct  = (pred_high - pred_low) / (close + 1e-12) * 100.0

    alpha   = 1.0 - nominal_coverage
    winkler = _winkler_score(pred_high, pred_low, act_c, alpha)

    return {
        "close_coverage_pct":    float(close_inside.mean() * 100.0),
        "envelope_coverage_pct": float(envelope_inside.mean() * 100.0),
        "mean_band_width_pct":   float(band_width_pct.mean()),
        f"p{int(sharpness_q*100)}_band_width_pct": float(np.percentile(band_width_pct, sharpness_q * 100)),
        "mae_high_price":        float(mean_absolute_error(act_h, pred_high)),
        "mae_low_price":         float(mean_absolute_error(act_l, pred_low)),
        "mean_winkler_score":    float(winkler),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pinball(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    residual = y_true - y_pred
    loss = np.where(residual >= 0, alpha * residual, (alpha - 1.0) * residual)
    return float(loss.mean())


def _winkler_score(
    upper: np.ndarray,
    lower: np.ndarray,
    actual: np.ndarray,
    alpha: float,
) -> float:
    """
    Winkler interval score (lower is better).
    Width + 2/α * penalty for points outside the interval.
    """
    width = upper - lower
    below = np.where(actual < lower, (lower - actual) * 2.0 / alpha, 0.0)
    above = np.where(actual > upper, (actual - upper) * 2.0 / alpha, 0.0)
    return float((width + below + above).mean())


# ---------------------------------------------------------------------------
# Persistence helpers used in pipeline
# ---------------------------------------------------------------------------

def save_eval_report(report: dict, cfg: dict) -> None:
    """Serialise the evaluation report to JSON."""
    art_dir = Path(cfg["artifacts"]["artifacts_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)
    out_path = art_dir / cfg["artifacts"]["eval_report"]
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Evaluation report saved to %s", out_path)


def save_predictions(
    test_df: pd.DataFrame,
    pred_upper_ret: np.ndarray,
    pred_lower_ret: np.ndarray,
    cfg: dict,
) -> None:
    """Save raw predictions alongside actuals as a parquet file."""
    art_dir = Path(cfg["artifacts"]["artifacts_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    close = test_df["close"].values
    band_upper = np.maximum(pred_upper_ret, pred_lower_ret)
    band_lower = np.minimum(pred_upper_ret, pred_lower_ret)

    out = test_df[["date", "ticker", "close",
                   "next_high_actual", "next_low_actual", "next_close_actual",
                   "target_upper", "target_lower"]].copy()
    out["pred_upper_ret"]  = pred_upper_ret
    out["pred_lower_ret"]  = pred_lower_ret
    out["pred_high_price"] = close * np.exp(band_upper)
    out["pred_low_price"]  = close * np.exp(band_lower)
    out["band_valid"]      = (out["pred_high_price"] >= out["pred_low_price"]).astype(int)

    out_path = art_dir / cfg["artifacts"]["predictions"]
    out.to_parquet(out_path, index=False)
    logger.info("Predictions saved to %s", out_path)
