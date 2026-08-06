"""
Model builders for the price-band forecasting pipeline.

Uses LightGBM quantile regression — the upper band model is fit at the
high quantile; the lower band model at the low quantile.  This gives us
calibrated prediction intervals rather than simple point forecasts.
"""


import logging
from typing import Any

import lightgbm as lgb

logger = logging.getLogger(__name__)


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

def build_model(
    params: dict[str, Any],
    quantile: float = 0.84,
    seed: int = 42,
) -> lgb.LGBMRegressor:
    """
    Build a LightGBM quantile regressor.

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
    return build_lgb_model(params, quantile=quantile, seed=seed)
