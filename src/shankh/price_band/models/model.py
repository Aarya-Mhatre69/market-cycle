"""
Model builders for the price-band forecasting pipeline.

Supports two backends:
- XGBoost  (quantile objective via  ``reg:quantileerror``)
- LightGBM (quantile objective via  ``quantile`` loss)

Both produce *quantile regressors* — the upper band model is fit at the
high quantile; the lower band model at the low quantile.  This gives us
calibrated prediction intervals rather than simple point forecasts.
"""


import logging
from typing import Any

import lightgbm as lgb
import xgboost as xgb

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XGBoost builder
# ---------------------------------------------------------------------------

def build_xgb_model(
    params: dict[str, Any],
    quantile: float = 0.84,
    seed: int = 42,
) -> xgb.XGBRegressor:
    """
    Build an XGBRegressor using the quantile regression objective.

    Parameters
    ----------
    params : dict
        Hyperparameters (n_estimators, learning_rate, etc.).
    quantile : float
        Target quantile level (0.84 → upper band, 0.16 → lower band).
    seed : int
        Random seed.

    Returns
    -------
    xgb.XGBRegressor
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    return xgb.XGBRegressor(
        n_estimators       = params.get("n_estimators", 1000),
        learning_rate      = params.get("learning_rate", 0.03),
        max_depth          = params.get("max_depth", 5),
        min_child_weight   = params.get("min_child_weight", 3),
        subsample          = params.get("subsample", 0.8),
        colsample_bytree   = params.get("colsample_bytree", 0.8),
        gamma              = params.get("gamma", 0.0),
        reg_alpha          = params.get("reg_alpha", 0.1),
        reg_lambda         = params.get("reg_lambda", 1.0),
        # Quantile objective available in XGBoost ≥ 1.7
        objective          = "reg:quantileerror",
        quantile_alpha     = quantile,
        tree_method        = params.get("tree_method", "hist"),
        eval_metric        = "quantile",
        random_state       = seed,
        n_jobs             = params.get("n_jobs", -1),
        early_stopping_rounds = params.get("early_stopping_rounds", 50),
    )


# ---------------------------------------------------------------------------
# LightGBM builder
# ---------------------------------------------------------------------------

def build_lgb_model(
    params: dict[str, Any],
    quantile: float = 0.84,
    seed: int = 42,
) -> lgb.LGBMRegressor:
    """
    Build an LGBMRegressor using the quantile regression objective
    (pinball loss).

    Parameters
    ----------
    params : dict
        Hyperparameters.
    quantile : float
        Target quantile level.
    seed : int
        Random seed.

    Returns
    -------
    lgb.LGBMRegressor
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    return lgb.LGBMRegressor(
        n_estimators       = params.get("n_estimators", 1000),
        learning_rate      = params.get("learning_rate", 0.03),
        num_leaves         = params.get("num_leaves", 63),
        max_depth          = params.get("max_depth", -1),
        min_child_samples  = params.get("min_child_samples", 20),
        subsample          = params.get("subsample", 0.8),
        subsample_freq     = 1,
        colsample_bytree   = params.get("colsample_bytree", 0.8),
        reg_alpha          = params.get("reg_alpha", 0.1),
        reg_lambda         = params.get("reg_lambda", 1.0),
        min_split_gain     = params.get("min_split_gain", 0.0),
        objective          = "quantile",
        alpha              = quantile,       # LightGBM's pinball α
        metric             = "quantile",
        random_state       = seed,
        n_jobs             = params.get("n_jobs", -1),
        verbose            = -1,
    )


# ---------------------------------------------------------------------------
# Generic factory
# ---------------------------------------------------------------------------

def build_model(
    backend: str,
    params: dict[str, Any],
    quantile: float = 0.84,
    seed: int = 42,
):
    """
    Dispatch to XGBoost or LightGBM builder.

    Parameters
    ----------
    backend : str
        "xgboost" or "lightgbm"
    params, quantile, seed : see per-backend builders.
    """
    backend = backend.lower().strip()
    if backend in ("xgboost", "xgb"):
        return build_xgb_model(params, quantile=quantile, seed=seed)
    if backend in ("lightgbm", "lgb", "lgbm"):
        return build_lgb_model(params, quantile=quantile, seed=seed)
    raise ValueError(
        f"Unknown backend '{backend}'. Choose 'xgboost' or 'lightgbm'."
    )
