"""
Training module for the price-band forecasting pipeline.

Responsibilities
----------------
- Hyperparameter optimisation with Optuna using proper walk-forward CV
  (pinball loss as the objective — the correct loss for quantile regression).
- Final model training on the full pre-test dataset with the best params.
- Per-fold CV metrics collection for diagnostics.
- Model and artifact serialisation.

No ticker identity is ever passed to the model.  Features are purely
OHLCV-derived signals; see features.py for the full feature catalogue.
"""


import json
import logging
import pickle
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from shankh.price_band.models.model import build_model
from shankh.price_band.validation import Fold, walk_forward_folds

logger = logging.getLogger(__name__)

# Silence Optuna's per-trial output unless debug logging is on
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Pinball (quantile) loss — the proper metric for interval forecasting
# ---------------------------------------------------------------------------

def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """
    Mean pinball (quantile) loss at level *alpha*.

    L(y, ŷ) = mean( alpha * max(y-ŷ, 0) + (1-alpha) * max(ŷ-y, 0) )
    """
    residual = y_true - y_pred
    loss = np.where(residual >= 0, alpha * residual, (alpha - 1.0) * residual)
    return float(loss.mean())


# ---------------------------------------------------------------------------
# Optuna objective factory
# ---------------------------------------------------------------------------

def _make_objective(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    quantile: float,
    folds: list[Fold],
    backend: str,
    search_space: dict,
    seed: int,
):
    """
    Return an Optuna objective that evaluates a hyperparameter set
    using average pinball loss across all walk-forward folds.
    """
    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, backend, search_space)
        fold_losses: list[float] = []

        for fold in folds:
            X_train = df.iloc[fold.train_idx][feature_cols]   # DataFrame, keeps feature names
            y_train = df.iloc[fold.train_idx][target_col].values
            X_val   = df.iloc[fold.val_idx][feature_cols]
            y_val   = df.iloc[fold.val_idx][target_col].values

            model = build_model(backend, params, quantile=quantile, seed=seed)
            _fit_model(model, backend, X_train, y_train, X_val, y_val, params)

            preds = model.predict(X_val)
            fold_losses.append(pinball_loss(y_val, preds, alpha=quantile))

            # Report intermediate value for Optuna pruning
            trial.report(np.mean(fold_losses), step=fold.fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_losses))

    return objective


def _sample_params(
    trial: optuna.Trial,
    backend: str,
    search_space: dict,
) -> dict[str, Any]:
    """Sample hyperparameters from the search space for a given backend."""
    params: dict[str, Any] = {}

    if backend in ("lightgbm", "lgb", "lgbm"):
        params["n_estimators"]       = trial.suggest_categorical("n_estimators", search_space["n_estimators"])
        params["learning_rate"]      = trial.suggest_float("learning_rate", *search_space["learning_rate"], log=True)
        params["num_leaves"]         = trial.suggest_categorical("num_leaves", search_space["num_leaves"])
        params["max_depth"]          = trial.suggest_categorical("max_depth", search_space["max_depth"])
        params["min_child_samples"]  = trial.suggest_categorical("min_child_samples", search_space["min_child_samples"])
        params["subsample"]          = trial.suggest_float("subsample", *search_space["subsample"])
        params["colsample_bytree"]   = trial.suggest_float("colsample_bytree", *search_space["colsample_bytree"])
        params["reg_alpha"]          = trial.suggest_float("reg_alpha", *search_space["reg_alpha"])
        params["reg_lambda"]         = trial.suggest_float("reg_lambda", *search_space["reg_lambda"])
        params["min_split_gain"]     = trial.suggest_float("min_split_gain", *search_space["min_split_gain"])
        params["early_stopping_rounds"] = trial.suggest_categorical(
            "early_stopping_rounds", search_space["early_stopping_rounds"]
        )

    else:  # xgboost
        params["n_estimators"]       = trial.suggest_categorical("n_estimators", search_space["n_estimators"])
        params["learning_rate"]      = trial.suggest_float("learning_rate", *search_space["learning_rate"], log=True)
        params["max_depth"]          = trial.suggest_categorical("max_depth", search_space["max_depth"])
        params["min_child_weight"]   = trial.suggest_categorical("min_child_weight", search_space["min_child_weight"])
        params["subsample"]          = trial.suggest_float("subsample", *search_space["subsample"])
        params["colsample_bytree"]   = trial.suggest_float("colsample_bytree", *search_space["colsample_bytree"])
        params["gamma"]              = trial.suggest_float("gamma", *search_space["gamma"])
        params["reg_alpha"]          = trial.suggest_float("reg_alpha", *search_space["reg_alpha"])
        params["reg_lambda"]         = trial.suggest_float("reg_lambda", *search_space["reg_lambda"])
        params["early_stopping_rounds"] = trial.suggest_categorical(
            "early_stopping_rounds", search_space["early_stopping_rounds"]
        )

    return params


# ---------------------------------------------------------------------------
# Fitting helper — handles early stopping API differences
# ---------------------------------------------------------------------------

def _fit_model(model, backend: str, X_train, y_train, X_val, y_val, params: dict) -> None:
    """Fit model with early stopping, handling XGBoost vs LightGBM APIs."""
    if backend in ("lightgbm", "lgb", "lgbm"):
        callbacks = [
            _lgb_early_stopping(params.get("early_stopping_rounds", 50)),
            _lgb_log_evaluation(-1),   # suppress per-iter output
        ]
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )
    else:  # xgboost
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )


def _lgb_early_stopping(rounds: int):
    """Return a LightGBM early-stopping callback."""
    import lightgbm as lgb
    return lgb.early_stopping(stopping_rounds=rounds, verbose=False)


def _lgb_log_evaluation(period: int):
    """Return a LightGBM log-evaluation callback."""
    import lightgbm as lgb
    return lgb.log_evaluation(period=period)


# ---------------------------------------------------------------------------
# Public: run Optuna HPO for one target (upper or lower band)
# ---------------------------------------------------------------------------

def run_hpo(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    quantile: float,
    folds: list[Fold],
    cfg: dict,
) -> tuple[optuna.Study, dict[str, Any]]:
    """
    Run Optuna HPO for one quantile target.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature-engineered frame (pre-test data only).
    feature_cols : list[str]
        Model input column names (no ticker, no targets).
    target_col : str
        Name of the target column ("target_upper" or "target_lower").
    quantile : float
        The quantile being modelled.
    folds : list[Fold]
        Walk-forward CV folds.
    cfg : dict
        Full CONFIG dict.

    Returns
    -------
    (study, best_params)
    """
    backend      = cfg["model"]["backend"]
    optuna_cfg   = cfg["optuna"]
    space_key    = f"{backend.replace('lightgbm','lgb').replace('xgboost','xgb')}_space"
    search_space = cfg["model"][space_key]
    seed         = cfg["seed"]

    pruner = _build_pruner(optuna_cfg)

    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
        sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)

    study = optuna.create_study(
        direction = optuna_cfg["direction"],
        sampler   = sampler,
        pruner    = pruner,
    )

    objective = _make_objective(
        df, feature_cols, target_col, quantile, folds, backend, search_space, seed
    )

    study.optimize(
        objective,
        n_trials  = optuna_cfg["n_trials"],
        timeout   = optuna_cfg.get("timeout"),
        n_jobs    = optuna_cfg.get("n_jobs", 1),
        show_progress_bar=False,
    )

    best_params = dict(study.best_trial.params)
    logger.info(
        "HPO [%s] done | best pinball=%.6f | trial=%d",
        target_col, study.best_value, study.best_trial.number,
    )
    return study, best_params


def _build_pruner(optuna_cfg: dict):
    pruner_name = optuna_cfg.get("pruner", "median")
    warmup      = optuna_cfg.get("n_warmup_steps", 10)
    if pruner_name == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=warmup)
    if pruner_name == "hyperband":
        return optuna.pruners.HyperbandPruner()
    return optuna.pruners.NopPruner()


# ---------------------------------------------------------------------------
# Public: train final models on full pre-test data
# ---------------------------------------------------------------------------

def train_final_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    best_upper_params: dict[str, Any],
    best_lower_params: dict[str, Any],
    cfg: dict,
) -> tuple[Any, Any]:
    """
    Train the final upper and lower band models on ALL pre-test rows.

    Uses a small chronological hold-out (last 10 % of pre-test dates)
    as the early-stopping validation set — this is NOT part of the
    Optuna CV folds, so there is no leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Pre-test feature frame (dates < test_cutoff).
    feature_cols : list[str]
    best_upper_params, best_lower_params : dict
        Best params from HPO.
    cfg : dict

    Returns
    -------
    (upper_model, lower_model)
    """
    backend         = cfg["model"]["backend"]
    upper_quantile  = cfg["model"]["upper_quantile"]
    lower_quantile  = cfg["model"]["lower_quantile"]
    seed            = cfg["seed"]

    # Chronological early-stopping split: last 10 % of unique dates
    all_dates = sorted(df["date"].unique())
    split_idx = max(1, int(len(all_dates) * 0.90))
    es_val_dates = set(all_dates[split_idx:])

    train_mask = ~df["date"].isin(es_val_dates)
    val_mask   =  df["date"].isin(es_val_dates)

    X_tr  = df.loc[train_mask, feature_cols]   # DataFrame — preserves feature names
    X_val = df.loc[val_mask,   feature_cols]

    upper_model = build_model(backend, best_upper_params, quantile=upper_quantile, seed=seed)
    lower_model = build_model(backend, best_lower_params, quantile=lower_quantile, seed=seed)

    logger.info("Training final upper-band model (%s q=%.2f) ...", backend, upper_quantile)
    _fit_model(
        upper_model, backend,
        X_tr,  df.loc[train_mask, "target_upper"].values,
        X_val, df.loc[val_mask,   "target_upper"].values,
        best_upper_params,
    )

    logger.info("Training final lower-band model (%s q=%.2f) ...", backend, lower_quantile)
    _fit_model(
        lower_model, backend,
        X_tr,  df.loc[train_mask, "target_lower"].values,
        X_val, df.loc[val_mask,   "target_lower"].values,
        best_lower_params,
    )

    return upper_model, lower_model


# ---------------------------------------------------------------------------
# Public: collect per-fold CV metrics for the final (best) params
# ---------------------------------------------------------------------------

def collect_cv_metrics(
    df: pd.DataFrame,
    feature_cols: list[str],
    best_upper_params: dict[str, Any],
    best_lower_params: dict[str, Any],
    folds: list[Fold],
    cfg: dict,
) -> list[dict]:
    """
    Re-train with best params on each fold and collect val metrics.

    Returns a list of per-fold metric dicts.
    """
    backend        = cfg["model"]["backend"]
    upper_quantile = cfg["model"]["upper_quantile"]
    lower_quantile = cfg["model"]["lower_quantile"]
    seed           = cfg["seed"]
    results        = []

    for fold in folds:
        X_tr  = df.iloc[fold.train_idx][feature_cols]   # DataFrame — preserves feature names
        X_val = df.iloc[fold.val_idx][feature_cols]
        y_u_tr  = df.iloc[fold.train_idx]["target_upper"].values
        y_l_tr  = df.iloc[fold.train_idx]["target_lower"].values
        y_u_val = df.iloc[fold.val_idx]["target_upper"].values
        y_l_val = df.iloc[fold.val_idx]["target_lower"].values

        u_model = build_model(backend, best_upper_params, quantile=upper_quantile, seed=seed)
        l_model = build_model(backend, best_lower_params, quantile=lower_quantile, seed=seed)

        _fit_model(u_model, backend, X_tr, y_u_tr, X_val, y_u_val, best_upper_params)
        _fit_model(l_model, backend, X_tr, y_l_tr, X_val, y_l_val, best_lower_params)

        pred_u = u_model.predict(X_val)
        pred_l = l_model.predict(X_val)

        val_df = df.iloc[fold.val_idx].copy()
        val_df["pred_upper"] = pred_u
        val_df["pred_lower"] = pred_l

        metrics = _fold_metrics(val_df, pred_u, pred_l, upper_quantile, lower_quantile)
        metrics["fold"]        = fold.fold_idx
        metrics["train_start"] = str(fold.train_start.date())
        metrics["train_end"]   = str(fold.train_end.date())
        metrics["val_start"]   = str(fold.val_start.date())
        metrics["val_end"]     = str(fold.val_end.date())
        metrics["n_train"]     = int(len(fold.train_idx))
        metrics["n_val"]       = int(len(fold.val_idx))
        results.append(metrics)

        logger.info(
            "Fold %d  val pinball_upper=%.5f  pinball_lower=%.5f  coverage=%.1f%%",
            fold.fold_idx,
            metrics["pinball_upper"],
            metrics["pinball_lower"],
            metrics["close_coverage_pct"],
        )

    return results


def _fold_metrics(
    val_df: pd.DataFrame,
    pred_upper: np.ndarray,
    pred_lower: np.ndarray,
    upper_quantile: float,
    lower_quantile: float,
) -> dict[str, float]:
    """Compute a standard set of interval-forecasting metrics for one fold."""
    # Enforce band ordering (lower ≤ upper)
    band_upper = np.maximum(pred_upper, pred_lower)
    band_lower = np.minimum(pred_upper, pred_lower)

    close  = val_df["close"].values
    act_h  = val_df["next_high_actual"].values
    act_l  = val_df["next_low_actual"].values
    act_c  = val_df["next_close_actual"].values

    pred_high_price = close * np.exp(band_upper)
    pred_low_price  = close * np.exp(band_lower)

    # Coverage: % of true next-day closes inside the predicted band
    close_inside = (act_c >= pred_low_price) & (act_c <= pred_high_price)
    # Envelope coverage: % of full [low, high] bar inside the band
    envelope_inside = (act_l >= pred_low_price) & (act_h <= pred_high_price)

    band_width_pct = (pred_high_price - pred_low_price) / (close + 1e-12)

    return {
        "pinball_upper": pinball_loss(val_df["target_upper"].values, pred_upper, upper_quantile),
        "pinball_lower": pinball_loss(val_df["target_lower"].values, pred_lower, lower_quantile),
        "mae_upper_ret": float(mean_absolute_error(val_df["target_upper"], pred_upper)),
        "mae_lower_ret": float(mean_absolute_error(val_df["target_lower"], pred_lower)),
        "rmse_upper_ret": float(np.sqrt(mean_squared_error(val_df["target_upper"], pred_upper))),
        "rmse_lower_ret": float(np.sqrt(mean_squared_error(val_df["target_lower"], pred_lower))),
        "mae_high_price": float(mean_absolute_error(act_h, pred_high_price)),
        "mae_low_price":  float(mean_absolute_error(act_l, pred_low_price)),
        "close_coverage_pct":    float(close_inside.mean() * 100.0),
        "envelope_coverage_pct": float(envelope_inside.mean() * 100.0),
        "mean_band_width_pct":   float(band_width_pct.mean() * 100.0),
        "p90_band_width_pct":    float(np.percentile(band_width_pct, 90) * 100.0),
    }


# ---------------------------------------------------------------------------
# Artifact serialisation
# ---------------------------------------------------------------------------

def save_artifacts(
    upper_model,
    lower_model,
    feature_cols: list[str],
    best_upper_params: dict,
    best_lower_params: dict,
    study_upper: optuna.Study,
    study_lower: optuna.Study,
    cv_results: list[dict],
    cfg: dict,
) -> None:
    """
    Persist all training artefacts:
    - trained models  (native format)
    - feature column list
    - best hyperparameters
    - Optuna studies
    - per-fold CV results
    """
    art_cfg  = cfg["artifacts"]
    backend  = cfg["model"]["backend"]
    art_dir  = Path(art_cfg["artifacts_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    # Models
    _save_model(upper_model, backend, art_dir / art_cfg["upper_model"])
    _save_model(lower_model, backend, art_dir / art_cfg["lower_model"])
    logger.info("Models saved to %s", art_dir)

    # Feature columns
    joblib.dump(feature_cols, art_dir / art_cfg["feature_cols"])

    # Best params
    with open(art_dir / art_cfg["optuna_best"], "w") as f:
        json.dump(
            {
                "upper": {
                    "trial": study_upper.best_trial.number,
                    "pinball_loss": study_upper.best_value,
                    "params": best_upper_params,
                },
                "lower": {
                    "trial": study_lower.best_trial.number,
                    "pinball_loss": study_lower.best_value,
                    "params": best_lower_params,
                },
                "backend": backend,
                "upper_quantile": cfg["model"]["upper_quantile"],
                "lower_quantile": cfg["model"]["lower_quantile"],
            },
            f, indent=2,
        )

    # Optuna studies (pickled for resume / analysis)
    with open(art_dir / art_cfg["optuna_study"].format(target="upper"), "wb") as f:
        pickle.dump(study_upper, f)
    with open(art_dir / art_cfg["optuna_study"].format(target="lower"), "wb") as f:
        pickle.dump(study_lower, f)

    # CV results
    with open(art_dir / art_cfg["cv_results"], "w") as f:
        json.dump(cv_results, f, indent=2)

    logger.info("All artifacts saved to %s", art_dir)


def save_feature_importance(upper_model, lower_model, feature_cols: list[str], cfg: dict) -> None:
    """Save feature importance tables (gain-based) for both models."""
    import pandas as pd
    art_dir = Path(cfg["artifacts"]["artifacts_dir"])

    for label, model in [("upper", upper_model), ("lower", lower_model)]:
        imp = _get_feature_importance(model, feature_cols, cfg["model"]["backend"])
        if imp is not None:
            out_path = art_dir / cfg["artifacts"]["feature_imp"].format(target=label)
            imp.to_csv(out_path, index=False)
            logger.info("Feature importance (%s) saved to %s", label, out_path)


def _get_feature_importance(model, feature_cols: list[str], backend: str):
    import pandas as pd
    try:
        if backend in ("lightgbm", "lgb", "lgbm"):
            imp_vals = model.booster_.feature_importance(importance_type="gain")
        else:
            imp_vals = model.feature_importances_
        df = pd.DataFrame({"feature": feature_cols, "importance": imp_vals})
        return df.sort_values("importance", ascending=False).reset_index(drop=True)
    except Exception as exc:
        logger.warning("Could not extract feature importance: %s", exc)
        return None


def _save_model(model, backend: str, base_path: Path) -> None:
    """Save model in its native format."""
    if backend in ("lightgbm", "lgb", "lgbm"):
        path = base_path.with_suffix(".txt")
        model.booster_.save_model(str(path))
    else:
        path = base_path.with_suffix(".json")
        model.save_model(str(path))
