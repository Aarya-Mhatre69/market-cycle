"""
Inference module for the price-band forecasting pipeline.

Given a trained upper-band model, a trained lower-band model, and a feature
frame for one or more instruments, produces validated price-band predictions.

Post-processing guarantees
--------------------------
1. band_lower ≤ band_upper  (ordering is enforced by sorting predictions)
2. Both price bands are strictly positive
3. Band width is at least a configurable minimum fraction of close price
   (prevents degenerate zero-width bands)

The inference path is instrument-agnostic — it accepts any feature frame
built by features.py and does not use ticker identity.
"""

import json
from typing import Tuple
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MIN_HALF_WIDTH_FRAC = 0.001   # 0.1 % of close




def predict_bands(
    df: pd.DataFrame,
    upper_model: Any,
    lower_model: Any,
    feature_cols: list[str],
    q_conf: float = 0.0,
) -> pd.DataFrame:
    """Run inference with optional conformal score adjustment."""
    missing = set(feature_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Feature frame is missing required columns: {missing}")

    X     = df[feature_cols]
    close = df["close"].values

    raw_upper = upper_model.predict(X) + q_conf
    raw_lower = lower_model.predict(X) - q_conf

    pred_upper, pred_lower = _enforce_band_ordering(raw_upper, raw_lower)
    pred_high  = close * np.exp(pred_upper)
    pred_low   = close * np.exp(pred_lower)

    out = df[["date", "ticker", "close"]].copy()
    out["pred_upper_ret"]  = pred_upper
    out["pred_lower_ret"]  = pred_lower
    out["pred_high_price"] = np.round(pred_high, 4)
    out["pred_low_price"]  = np.round(pred_low,  4)
    out["band_width_pct"]  = (pred_high - pred_low) / (close + 1e-12) * 100.0
    out["band_ordering_ok"] = (out["pred_high_price"] >= out["pred_low_price"]).astype(int)

    return out


def load_models_and_features(cfg: dict) -> Tuple[Any, Any, list[str], float]:
    """Load trained boosters, feature columns list, and conformal score adjustment."""
    art_dir = Path(cfg["artifacts"]["artifacts_dir"])
    art_cfg = cfg["artifacts"]

    upper_model = _load_model(art_dir / art_cfg["upper_model"])
    lower_model = _load_model(art_dir / art_cfg["lower_model"])
    feature_cols: list[str] = joblib.load(art_dir / art_cfg["feature_cols"])

    q_conf = 0.0
    conf_path = art_dir / art_cfg.get("conformal_score", "conformal_score.json")
    if conf_path.exists():
        with open(conf_path, "r") as f:
            conf_data = json.load(f)
            q_conf = float(conf_data.get("q_conf", 0.0))

    logger.info("Loaded models, %d features, and q_conf=%.5f from %s", len(feature_cols), q_conf, art_dir)
    return upper_model, lower_model, feature_cols, q_conf


def _load_model(base_path: Path) -> Any:
    import lightgbm as lgb
    path = base_path.with_suffix(".txt")
    return lgb.Booster(model_file=str(path))


def predict_next_day_band(
    row: dict[str, float],
    upper_model: Any,
    lower_model: Any,
    feature_cols: list[str],
    close: float,
) -> dict[str, float]:
    """
    Predict the next-day price band for a single observation.

    Parameters
    ----------
    row : dict
        Feature values keyed by feature name.
    upper_model, lower_model : fitted models
    feature_cols : list[str]
    close : float
        Today's close price (used to convert log-return to price).

    Returns
    -------
    dict with keys: pred_upper_ret, pred_lower_ret,
                    pred_high_price, pred_low_price, band_width_pct
    """
    import pandas as pd
    x = pd.DataFrame([[row[c] for c in feature_cols]], columns=feature_cols)
    raw_upper = float(upper_model.predict(x)[0])
    raw_lower = float(lower_model.predict(x)[0])

    upper_ret, lower_ret = _enforce_band_ordering(
        np.array([raw_upper]), np.array([raw_lower])
    )
    upper_ret = float(upper_ret[0])
    lower_ret = float(lower_ret[0])

    pred_high = round(close * np.exp(upper_ret), 4)
    pred_low  = round(close * np.exp(lower_ret), 4)

    return {
        "pred_upper_ret":  upper_ret,
        "pred_lower_ret":  lower_ret,
        "pred_high_price": pred_high,
        "pred_low_price":  pred_low,
        "band_width_pct":  (pred_high - pred_low) / (close + 1e-12) * 100.0,
    }


def _load_model(base_path: Path) -> Any:
    import lightgbm as lgb
    path = base_path.with_suffix(".txt")
    return lgb.Booster(model_file=str(path))


def _enforce_band_ordering(
    raw_upper: np.ndarray,
    raw_lower: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Guarantee that band_upper ≥ band_lower and that the band has a
    minimum width of _MIN_HALF_WIDTH_FRAC of the close price.

    Strategy
    --------
    1. Sort so upper ≥ lower (swap where inverted).
    2. If band width in price space < minimum threshold, expand symmetrically
       around the midpoint.
    """
    upper = np.maximum(raw_upper, raw_lower)
    lower = np.minimum(raw_upper, raw_lower)

    # Minimum band half-width in log-return space (≈ fraction for small values)
    min_half = _MIN_HALF_WIDTH_FRAC
    mid      = (upper + lower) / 2.0
    half     = (upper - lower) / 2.0
    half_adj = np.maximum(half, min_half)

    upper = mid + half_adj
    lower = mid - half_adj

    return upper, lower
