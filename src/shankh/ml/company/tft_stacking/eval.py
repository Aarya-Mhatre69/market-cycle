"""
Evaluation module computing interval accuracy, coverage, and Winkler metrics for TFT.
"""
import logging
from typing import Dict, Any
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

logger = logging.getLogger(__name__)


def evaluate_tft(
    model: TemporalFusionTransformer,
    test_ds: TimeSeriesDataSet,
    test_df: pd.DataFrame,
    cfg: dict,
) -> Dict[str, Any]:
    """
    Runs multi-quantile inference on the test set and returns comprehensive interval metrics.
    """
    test_dataloader = test_ds.to_dataloader(train=False, batch_size=128, num_workers=2)

    # Output shape: (N_samples, prediction_horizon=1, n_quantiles=3)
    predictions = model.predict(test_dataloader, mode="quantiles", return_x=False)
    
    pred_lower_ret  = predictions[:, 0, 0].cpu().numpy()  # alpha = 0.16 (Index 0)
    pred_median_ret = predictions[:, 0, 1].cpu().numpy()  # alpha = 0.50 (Index 1)
    pred_upper_ret  = predictions[:, 0, 2].cpu().numpy()  # alpha = 0.84 (Index 2)

    # Post-processing: Enforce upper >= lower
    band_upper = np.maximum(pred_upper_ret, pred_lower_ret)
    band_lower = np.minimum(pred_upper_ret, pred_lower_ret)

    n_eval = len(band_upper)
    test_slice = test_df.iloc[:n_eval].copy()

    close = test_slice["close"].values
    
    if "next_high_actual" in test_slice.columns:
        act_h = test_slice["next_high_actual"].values
        act_l = test_slice["next_low_actual"].values
        act_c = test_slice["next_close_actual"].values
    else:
        act_c = close * np.exp(test_slice["target_return"].values)
        act_h = act_c
        act_l = act_c

    pred_high = close * np.exp(band_upper)
    pred_low  = close * np.exp(band_lower)

    close_inside    = (act_c >= pred_low) & (act_c <= pred_high)
    envelope_inside = (act_l >= pred_low) & (act_h <= pred_high)
    band_width_pct  = (pred_high - pred_low) / (close + 1e-12) * 100.0

    nominal_coverage = cfg["evaluation"]["target_coverage"]
    empirical_coverage = float(close_inside.mean())
    calibration_error  = abs(empirical_coverage - nominal_coverage)

    alpha = 1.0 - nominal_coverage
    width = pred_high - pred_low
    below = np.where(act_c < pred_low, (pred_low - act_c) * 2.0 / alpha, 0.0)
    above = np.where(act_c > pred_high, (act_c - pred_high) * 2.0 / alpha, 0.0)
    winkler_score = float((width + below + above).mean())

    metrics = {
        "close_coverage_pct": float(close_inside.mean() * 100.0),
        "envelope_coverage_pct": float(envelope_inside.mean() * 100.0),
        "nominal_coverage_pct": float(nominal_coverage * 100.0),
        "calibration_error": float(calibration_error),
        "mean_band_width_pct": float(band_width_pct.mean()),
        "median_band_width_pct": float(np.median(band_width_pct)),
        "p90_band_width_pct": float(np.percentile(band_width_pct, 90)),
        "mae_high_price": float(np.abs(act_h - pred_high).mean()),
        "mae_low_price": float(np.abs(act_l - pred_low).mean()),
        "mean_winkler_score": winkler_score,
        "n_samples": int(n_eval),
    }

    logger.info(
        "TFT Evaluation Complete | Close Coverage: %.2f%% | Band Width: %.2f%% | Winkler: %.2f",
        metrics["close_coverage_pct"],
        metrics["mean_band_width_pct"],
        metrics["mean_winkler_score"],
    )

    return metrics