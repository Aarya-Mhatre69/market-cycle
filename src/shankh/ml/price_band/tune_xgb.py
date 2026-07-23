"""
Full Productionized Training & Evaluation Pipeline for Price-Band Forecasting.

Merges:
- Multi-backend Quantile Regression Builders (LightGBM & XGBoost)
- Walk-Forward Cross-Validation Splitter (Date-aligned, leak-free)
- Optuna Hyperparameter Optimization (Pinball loss minimization with pruning)
- Chronological Early-Stopping Final Model Training
- Per-Fold CV Metrics & Gain-Based Feature Importance Persistences
- Test Set Evaluation against Persistence & Volatility Baselines
- Matplotlib Prediction Visualization & CLI Commands

Usage
-----
# Full pipeline run with Optuna HPO (LightGBM):
$ python train_pipeline.py --backend lightgbm --n-trials 30

# Full pipeline run with XGBoost:
$ python train_pipeline.py --backend xgboost --n-trials 40

# Fast execution without HPO (using baseline config parameters):
$ python train_pipeline.py --skip-hpo

# Plot predictions directly from saved Parquet without retraining:
$ python train_pipeline.py --plot-only --ticker INFY.NS
"""

import argparse
import copy
import json
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb
import xgboost as xgb

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.shankh.ml.price_band.config import CONFIG
from src.shankh.ml.price_band.data_loader import load_ohlcv
from src.shankh.ml.price_band.evaluation import evaluate, save_eval_report, save_predictions
from src.shankh.ml.price_band.features import add_features, get_feature_cols
from src.shankh.ml.price_band.inference import predict_bands
from src.shankh.ml.price_band.validation import Fold, walk_forward_folds

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("price_band_pipeline")
optuna.logging.set_verbosity(optuna.logging.WARNING)



def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_config(override_path: Optional[str] = None) -> dict:
    cfg = copy.deepcopy(CONFIG)
    if override_path:
        with open(override_path, "r") as f:
            overrides = json.load(f)
        cfg = _deep_merge(cfg, overrides)
        logger.info("Config overrides loaded from %s", override_path)
    return cfg



def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Mean pinball (quantile) loss at quantile level alpha."""
    residual = y_true - y_pred
    loss = np.where(residual >= 0, alpha * residual, (alpha - 1.0) * residual)
    return float(loss.mean())



def build_xgb_model(params: dict[str, Any], quantile: float = 0.84, seed: int = 42) -> xgb.XGBRegressor:
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    return xgb.XGBRegressor(
        n_estimators=params.get("n_estimators", 1000),
        learning_rate=params.get("learning_rate", 0.03),
        max_depth=params.get("max_depth", 5),
        min_child_weight=params.get("min_child_weight", 3),
        subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample_bytree", 0.8),
        gamma=params.get("gamma", 0.0),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
        objective="reg:quantileerror",
        quantile_alpha=quantile,
        tree_method=params.get("tree_method", "hist"),
        eval_metric="quantile",
        random_state=seed,
        n_jobs=params.get("n_jobs", -1),
    )


def build_lgb_model(params: dict[str, Any], quantile: float = 0.84, seed: int = 42) -> lgb.LGBMRegressor:
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")

    return lgb.LGBMRegressor(
        n_estimators=params.get("n_estimators", 1000),
        learning_rate=params.get("learning_rate", 0.03),
        num_leaves=params.get("num_leaves", 63),
        max_depth=params.get("max_depth", -1),
        min_child_samples=params.get("min_child_samples", 20),
        subsample=params.get("subsample", 0.8),
        subsample_freq=1,
        colsample_bytree=params.get("colsample_bytree", 0.8),
        reg_alpha=params.get("reg_alpha", 0.1),
        reg_lambda=params.get("reg_lambda", 1.0),
        min_split_gain=params.get("min_split_gain", 0.0),
        objective="quantile",
        alpha=quantile,
        metric="quantile",
        random_state=seed,
        n_jobs=params.get("n_jobs", -1),
        verbose=-1,
    )


def build_model(backend: str, params: dict[str, Any], quantile: float = 0.84, seed: int = 42):
    backend = backend.lower().strip()
    if backend in ("xgboost", "xgb"):
        return build_xgb_model(params, quantile=quantile, seed=seed)
    if backend in ("lightgbm", "lgb", "lgbm"):
        return build_lgb_model(params, quantile=quantile, seed=seed)
    raise ValueError(f"Unknown backend '{backend}'. Choose 'xgboost' or 'lightgbm'.")


def _fit_model(model, backend: str, X_train, y_train, X_val, y_val, params: dict) -> None:
    es_rounds = params.get("early_stopping_rounds", 50)
    if backend in ("lightgbm", "lgb", "lgbm"):
        callbacks = [
            lgb.early_stopping(stopping_rounds=es_rounds, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=callbacks)
    else:  # xgboost
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)



def _sample_params(trial: optuna.Trial, backend: str, search_space: dict) -> dict[str, Any]:
    params: dict[str, Any] = {}
    if backend in ("lightgbm", "lgb", "lgbm"):
        params["n_estimators"]      = trial.suggest_categorical("n_estimators", search_space["n_estimators"])
        params["learning_rate"]     = trial.suggest_float("learning_rate", *search_space["learning_rate"], log=True)
        params["num_leaves"]        = trial.suggest_categorical("num_leaves", search_space["num_leaves"])
        params["max_depth"]         = trial.suggest_categorical("max_depth", search_space["max_depth"])
        params["min_child_samples"] = trial.suggest_categorical("min_child_samples", search_space["min_child_samples"])
        params["subsample"]         = trial.suggest_float("subsample", *search_space["subsample"])
        params["colsample_bytree"]  = trial.suggest_float("colsample_bytree", *search_space["colsample_bytree"])
        params["reg_alpha"]         = trial.suggest_float("reg_alpha", *search_space["reg_alpha"])
        params["reg_lambda"]        = trial.suggest_float("reg_lambda", *search_space["reg_lambda"])
        params["min_split_gain"]    = trial.suggest_float("min_split_gain", *search_space["min_split_gain"])
        params["early_stopping_rounds"] = trial.suggest_categorical("early_stopping_rounds", search_space["early_stopping_rounds"])
    else:  # xgboost
        params["n_estimators"]      = trial.suggest_categorical("n_estimators", search_space["n_estimators"])
        params["learning_rate"]     = trial.suggest_float("learning_rate", *search_space["learning_rate"], log=True)
        params["max_depth"]         = trial.suggest_categorical("max_depth", search_space["max_depth"])
        params["min_child_weight"]  = trial.suggest_categorical("min_child_weight", search_space["min_child_weight"])
        params["subsample"]         = trial.suggest_float("subsample", *search_space["subsample"])
        params["colsample_bytree"]  = trial.suggest_float("colsample_bytree", *search_space["colsample_bytree"])
        params["gamma"]             = trial.suggest_float("gamma", *search_space["gamma"])
        params["reg_alpha"]         = trial.suggest_float("reg_alpha", *search_space["reg_alpha"])
        params["reg_lambda"]        = trial.suggest_float("reg_lambda", *search_space["reg_lambda"])
        params["early_stopping_rounds"] = trial.suggest_categorical("early_stopping_rounds", search_space["early_stopping_rounds"])
    return params


def run_hpo(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    quantile: float,
    folds: List[Fold],
    cfg: dict,
) -> Tuple[optuna.Study, dict[str, Any]]:
    backend      = cfg["model"]["backend"]
    optuna_cfg   = cfg["optuna"]
    space_key    = f"{backend.replace('lightgbm','lgb').replace('xgboost','xgb')}_space"
    search_space = cfg["model"][space_key]
    seed         = cfg["seed"]

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=optuna_cfg.get("n_warmup_steps", 10))
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True)

    study = optuna.create_study(direction=optuna_cfg["direction"], sampler=sampler, pruner=pruner)

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, backend, search_space)
        fold_losses: List[float] = []

        for fold in folds:
            X_train = df.iloc[fold.train_idx][feature_cols]
            y_train = df.iloc[fold.train_idx][target_col].values
            X_val   = df.iloc[fold.val_idx][feature_cols]
            y_val   = df.iloc[fold.val_idx][target_col].values

            model = build_model(backend, params, quantile=quantile, seed=seed)
            _fit_model(model, backend, X_train, y_train, X_val, y_val, params)

            preds = model.predict(X_val)
            fold_losses.append(pinball_loss(y_val, preds, alpha=quantile))

            trial.report(float(np.mean(fold_losses)), step=fold.fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(fold_losses))

    study.optimize(
        objective,
        n_trials=optuna_cfg["n_trials"],
        timeout=optuna_cfg.get("timeout"),
        n_jobs=optuna_cfg.get("n_jobs", 1),
        show_progress_bar=True,
    )

    best_params = dict(study.best_trial.params)
    logger.info("HPO [%s] complete | Best Pinball Loss = %.6f | Best Trial #%d", target_col, study.best_value, study.best_trial.number)
    return study, best_params



def train_final_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    best_upper_params: dict[str, Any],
    best_lower_params: dict[str, Any],
    cfg: dict,
) -> Tuple[Any, Any]:
    backend        = cfg["model"]["backend"]
    upper_quantile = cfg["model"]["upper_quantile"]
    lower_quantile = cfg["model"]["lower_quantile"]
    seed           = cfg["seed"]

    # Chronological holdout: last 10% of unique pre-test dates for early stopping
    all_dates = sorted(df["date"].unique())
    split_idx = max(1, int(len(all_dates) * 0.90))
    es_val_dates = set(all_dates[split_idx:])

    train_mask = ~df["date"].isin(es_val_dates)
    val_mask   = df["date"].isin(es_val_dates)

    X_tr  = df.loc[train_mask, feature_cols]
    X_val = df.loc[val_mask, feature_cols]

    upper_model = build_model(backend, best_upper_params, quantile=upper_quantile, seed=seed)
    lower_model = build_model(backend, best_lower_params, quantile=lower_quantile, seed=seed)

    logger.info("Fitting final upper-band model (%s, q=%.2f) ...", backend, upper_quantile)
    _fit_model(upper_model, backend, X_tr, df.loc[train_mask, cfg["targets"]["upper"]].values, X_val, df.loc[val_mask, cfg["targets"]["upper"]].values, best_upper_params)

    logger.info("Fitting final lower-band model (%s, q=%.2f) ...", backend, lower_quantile)
    _fit_model(lower_model, backend, X_tr, df.loc[train_mask, cfg["targets"]["lower"]].values, X_val, df.loc[val_mask, cfg["targets"]["lower"]].values, best_lower_params)

    return upper_model, lower_model


def collect_cv_metrics(
    df: pd.DataFrame,
    feature_cols: List[str],
    best_upper_params: dict[str, Any],
    best_lower_params: dict[str, Any],
    folds: List[Fold],
    cfg: dict,
) -> List[dict]:
    backend        = cfg["model"]["backend"]
    upper_quantile = cfg["model"]["upper_quantile"]
    lower_quantile = cfg["model"]["lower_quantile"]
    seed           = cfg["seed"]
    results        = []

    for fold in folds:
        X_tr  = df.iloc[fold.train_idx][feature_cols]
        X_val = df.iloc[fold.val_idx][feature_cols]
        y_u_tr, y_l_tr   = df.iloc[fold.train_idx][cfg["targets"]["upper"]].values, df.iloc[fold.train_idx][cfg["targets"]["lower"]].values
        y_u_val, y_l_val = df.iloc[fold.val_idx][cfg["targets"]["upper"]].values, df.iloc[fold.val_idx][cfg["targets"]["lower"]].values

        u_model = build_model(backend, best_upper_params, quantile=upper_quantile, seed=seed)
        l_model = build_model(backend, best_lower_params, quantile=lower_quantile, seed=seed)

        _fit_model(u_model, backend, X_tr, y_u_tr, X_val, y_u_val, best_upper_params)
        _fit_model(l_model, backend, X_tr, y_l_tr, X_val, y_l_val, best_lower_params)

        pred_u = u_model.predict(X_val)
        pred_l = l_model.predict(X_val)

        val_df = df.iloc[fold.val_idx].copy()
        
        # Enforce band ordering & compute metrics
        band_upper = np.maximum(pred_u, pred_l)
        band_lower = np.minimum(pred_u, pred_l)
        close = val_df["close"].values
        pred_high = close * np.exp(band_upper)
        pred_low  = close * np.exp(band_lower)

        close_inside = (val_df["next_close_actual"].values >= pred_low) & (val_df["next_close_actual"].values <= pred_high)
        envelope_inside = (val_df["next_low_actual"].values >= pred_low) & (val_df["next_high_actual"].values <= pred_high)
        width_pct = (pred_high - pred_low) / (close + 1e-12) * 100.0

        metrics = {
            "fold": fold.fold_idx,
            "train_start": str(fold.train_start.date()),
            "train_end": str(fold.train_end.date()),
            "val_start": str(fold.val_start.date()),
            "val_end": str(fold.val_end.date()),
            "pinball_upper": pinball_loss(y_u_val, pred_u, upper_quantile),
            "pinball_lower": pinball_loss(y_l_val, pred_l, lower_quantile),
            "close_coverage_pct": float(close_inside.mean() * 100.0),
            "envelope_coverage_pct": float(envelope_inside.mean() * 100.0),
            "mean_band_width_pct": float(width_pct.mean()),
        }
        results.append(metrics)
        logger.info("Fold %d Validation | Pinball Upper=%.5f | Pinball Lower=%.5f | Coverage=%.1f%%", fold.fold_idx, metrics["pinball_upper"], metrics["pinball_lower"], metrics["close_coverage_pct"])

    return results



def save_artifacts(
    upper_model, lower_model, feature_cols: List[str],
    best_upper_params: dict, best_lower_params: dict,
    study_upper: Optional[optuna.Study], study_lower: Optional[optuna.Study],
    cv_results: List[dict], cfg: dict
) -> None:
    art_cfg = cfg["artifacts"]
    backend = cfg["model"]["backend"]
    art_dir = Path(art_cfg["artifacts_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    # Save Native Model File (.txt for LightGBM, .json for XGBoost)
    if backend in ("lightgbm", "lgb", "lgbm"):
        upper_model.booster_.save_model(str((art_dir / art_cfg["upper_model"]).with_suffix(".txt")))
        lower_model.booster_.save_model(str((art_dir / art_cfg["lower_model"]).with_suffix(".txt")))
    else:
        upper_model.save_model(str((art_dir / art_cfg["upper_model"]).with_suffix(".json")))
        lower_model.save_model(str((art_dir / art_cfg["lower_model"]).with_suffix(".json")))

    # Feature columns list
    joblib.dump(feature_cols, art_dir / art_cfg["feature_cols"])

    # Optuna Best Parameters JSON
    with open(art_dir / art_cfg["optuna_best"], "w") as f:
        json.dump({
            "backend": backend,
            "upper": {"params": best_upper_params, "pinball": study_upper.best_value if study_upper else None},
            "lower": {"params": best_lower_params, "pinball": study_lower.best_value if study_lower else None},
        }, f, indent=2)

    # Pickled Studies & CV JSON
    if study_upper:
        with open(art_dir / art_cfg["optuna_study"].format(target="upper"), "wb") as f:
            pickle.dump(study_upper, f)
    if study_lower:
        with open(art_dir / art_cfg["optuna_study"].format(target="lower"), "wb") as f:
            pickle.dump(study_lower, f)

    with open(art_dir / art_cfg["cv_results"], "w") as f:
        json.dump(cv_results, f, indent=2)

    logger.info("All pipeline artifacts successfully saved to %s", art_dir)


def save_feature_importance(upper_model, lower_model, feature_cols: List[str], cfg: dict) -> None:
    art_dir = Path(cfg["artifacts"]["artifacts_dir"])
    backend = cfg["model"]["backend"]

    for label, model in [("upper", upper_model), ("lower", lower_model)]:
        try:
            if backend in ("lightgbm", "lgb", "lgbm"):
                imp_vals = model.booster_.feature_importance(importance_type="gain")
            else:
                imp_vals = model.feature_importances_
            imp_df = pd.DataFrame({"feature": feature_cols, "importance": imp_vals}).sort_values("importance", ascending=False).reset_index(drop=True)
            out_path = art_dir / cfg["artifacts"]["feature_imp"].format(target=label)
            imp_df.to_csv(out_path, index=False)
            logger.info("Saved feature importance (%s) to %s", label, out_path)
        except Exception as exc:
            logger.warning("Could not save feature importance for %s: %s", label, exc)



def load_predictions(cfg: Optional[dict] = None) -> pd.DataFrame:
    cfg = cfg or CONFIG
    art_dir = Path(cfg["artifacts"]["artifacts_dir"])
    pred_path = art_dir / cfg["artifacts"]["predictions"]
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found at {pred_path}. Run full pipeline first.")
    return pd.read_parquet(pred_path)


def plot_predictions(
    df: pd.DataFrame,
    ticker: Optional[str] = None,
    out_path: Optional[Path | str] = None,
    show: bool = False,
    sample_every: int = 3,
) -> None:
    work = df.copy()
    if ticker is not None:
        work = work[work["ticker"] == ticker]
    else:
        unique_tickers = sorted(work["ticker"].dropna().astype(str).unique())
        if unique_tickers:
            ticker = unique_tickers[0]
            logger.info("Multiple tickers found. Plotting %s. Use --ticker to select another.", ticker)
            work = work[work["ticker"] == ticker]

    if work.empty:
        raise ValueError(f"No rows found to plot for ticker {ticker!r}")

    work = work.sort_values("date").copy()
    work["date"] = pd.to_datetime(work["date"])

    if len(work) > 300:
        work = work.iloc[:: max(1, sample_every)].copy()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(work["date"], work["close"], label="Close", color="black", linewidth=1.5)
    ax.fill_between(
        work["date"],
        work["pred_low_price"],
        work["pred_high_price"],
        color="tab:blue",
        alpha=0.20,
        label="Predicted Price Band",
    )
    ax.plot(work["date"], work["next_high_actual"], color="green", linewidth=0.9, alpha=0.8, label="Actual Next High")
    ax.plot(work["date"], work["next_low_actual"], color="red", linewidth=0.9, alpha=0.8, label="Actual Next Low")
    ax.plot(work["date"], work["next_close_actual"], linestyle="--", color="gray", linewidth=1.0, label="Actual Next Close")

    ax.set_title(f"Price-Band Forecasting ({ticker or 'Portfolio'})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (INR)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()

    out_path = Path(out_path) if out_path else Path("prediction_plot.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    logger.info("Saved prediction plot to %s", out_path)
    if show:
        plt.show()
    plt.close(fig)



def run_pipeline(cfg: dict, skip_hpo: bool = False) -> dict:
    _set_seed(cfg["seed"])

    logger.info("=" * 70)
    logger.info("STEP 1 — Ingesting and aligning OHLCV panel")
    raw_df = load_ohlcv(cfg["data"]["data_dir"], required_cols=cfg["data"]["required_cols"])
    logger.info("Ingested %d total rows | %d tickers | %s to %s", len(raw_df), raw_df["ticker"].nunique(), raw_df["date"].min().date(), raw_df["date"].max().date())

    logger.info("STEP 2 — Feature engineering and target calculation")
    feat_df = add_features(raw_df, cfg=cfg["features"])
    feature_cols = get_feature_cols(feat_df)
    logger.info("Engineered feature set: %d rows × %d features", len(feat_df), len(feature_cols))

    logger.info("STEP 3 — Train / Test set partitioning (Test cutoff: %s)", cfg["data"]["test_cutoff"])
    test_cutoff = cfg["data"]["test_cutoff"]
    pretrain_df = feat_df[feat_df["date"] < test_cutoff].copy().reset_index(drop=True)
    test_df     = feat_df[feat_df["date"] >= test_cutoff].copy().reset_index(drop=True)

    if pretrain_df.empty or test_df.empty:
        raise ValueError(f"Partition empty at cutoff {test_cutoff}. Verify dataset coverage.")

    logger.info("STEP 4 — Generating expanding-window Walk-Forward CV folds")
    folds = walk_forward_folds(
        pretrain_df,
        n_splits=cfg["walk_forward"]["n_splits"],
        val_size=cfg["walk_forward"]["val_size"],
        min_train_size=cfg["walk_forward"]["min_train_size"],
        gap=cfg["walk_forward"]["gap"],
    )

    study_upper, study_lower = None, None
    if not skip_hpo:
        logger.info("STEP 5 — Executing Optuna HPO (%d trials, backend: %s)", cfg["optuna"]["n_trials"], cfg["model"]["backend"])
        study_upper, best_upper_params = run_hpo(pretrain_df, feature_cols, cfg["targets"]["upper"], cfg["model"]["upper_quantile"], folds, cfg)
        study_lower, best_lower_params = run_hpo(pretrain_df, feature_cols, cfg["targets"]["lower"], cfg["model"]["lower_quantile"], folds, cfg)
    else:
        logger.info("STEP 5 — Skipping Optuna HPO; using default baseline search space parameters")
        backend = cfg["model"]["backend"]
        space_key = f"{backend.replace('lightgbm','lgb').replace('xgboost','xgb')}_space"
        defaults = {k: v[0] if isinstance(v, list) else v[0] for k, v in cfg["model"][space_key].items()}
        best_upper_params, best_lower_params = defaults, defaults

    logger.info("STEP 6 — Training final models on pre-test set with chronological holdout")
    upper_model, lower_model = train_final_models(pretrain_df, feature_cols, best_upper_params, best_lower_params, cfg)

    logger.info("STEP 7 — Collecting walk-forward CV diagnostic metrics")
    cv_results = collect_cv_metrics(pretrain_df, feature_cols, best_upper_params, best_lower_params, folds, cfg)

    logger.info("STEP 8 — Persisting models, configuration, and feature importance artifacts")
    save_artifacts(upper_model, lower_model, feature_cols, best_upper_params, best_lower_params, study_upper, study_lower, cv_results, cfg)
    save_feature_importance(upper_model, lower_model, feature_cols, cfg)

    logger.info("STEP 9 — Inference on held-out test set")
    pred_df = predict_bands(test_df, upper_model, lower_model, feature_cols)

    logger.info("STEP 10 — Evaluating test predictions against baselines")
    report = evaluate(test_df, pred_df["pred_upper_ret"].values, pred_df["pred_lower_ret"].values, cfg)

    logger.info("STEP 11 — Saving test evaluation outputs & rendering visualization")
    save_eval_report(report, cfg)
    save_predictions(test_df, pred_df["pred_upper_ret"].values, pred_df["pred_lower_ret"].values, cfg)

    try:
        preds_df = load_predictions(cfg)
        out_plot_path = Path(cfg["artifacts"]["artifacts_dir"]) / f"prediction_plot_{cfg['model']['backend']}.png"
        plot_predictions(preds_df, out_path=out_plot_path)
    except Exception as exc:
        logger.warning("Could not render prediction plot: %s", exc)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE. Artifacts stored in: %s", cfg["artifacts"]["artifacts_dir"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Full Productionized Price-Band Forecasting Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON file with config overrides.")
    parser.add_argument("--backend", type=str, choices=["lightgbm", "xgboost"], default=None, help="Override backend model choice.")
    parser.add_argument("--n-trials", type=int, default=None, help="Override number of Optuna HPO trials.")
    parser.add_argument("--skip-hpo", action="store_true", help="Skip Optuna HPO and fit baseline parameters quickly.")
    parser.add_argument("--plot-only", action="store_true", help="Render plot directly from saved predictions without retraining.")
    parser.add_argument("--ticker", type=str, default=None, help="Specific ticker to plot when using --plot-only.")
    args = parser.parse_args()

    cfg = _load_config(args.config)

    if args.backend:
        cfg["model"]["backend"] = args.backend
    if args.n_trials:
        cfg["optuna"]["n_trials"] = args.n_trials

    if args.plot_only:
        logger.info("Plot-only mode activated. Loading saved Parquet table...")
        preds_df = load_predictions(cfg)
        out_path = Path(cfg["artifacts"]["artifacts_dir"]) / f"prediction_plot_{cfg['model']['backend']}.png"
        plot_predictions(preds_df, ticker=args.ticker, out_path=out_path, show=True)
    else:
        run_pipeline(cfg, skip_hpo=args.skip_hpo)


if __name__ == "__main__":
    main()