"""
Top-level pipeline orchestrator for price-band forecasting.

Execution order
---------------
1. Load and validate OHLCV data from all CSV files
2. Engineer features (purely OHLCV-derived, no ticker identity)
3. Split into pre-test (CV + training) and held-out test sets
4. Generate walk-forward CV folds from the pre-test set
5. Run Optuna HPO for upper band (pinball loss @ upper_quantile)
6. Run Optuna HPO for lower band (pinball loss @ lower_quantile)
7. Collect per-fold CV metrics with best params (diagnostics)
8. Train final models on full pre-test data
9. Run inference on held-out test set
10. Full evaluation: calibration, sharpness, accuracy, baselines
11. Save all artifacts

Run with:
    python -m shankh.price_band.pipeline
    python -m shankh.price_band.pipeline --config path/to/override.json
"""


import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import numpy as np

from shankh.price_band.config import CONFIG
from shankh.price_band.data_loader import load_ohlcv
from shankh.price_band.evaluation import (
    evaluate,
    save_eval_report,
    save_predictions,
)
from shankh.price_band.features import add_features, get_feature_cols
from shankh.price_band.inference import predict_bands
from shankh.price_band.models.trainer import (
    collect_cv_metrics,
    run_hpo,
    save_artifacts,
    save_feature_importance,
    train_final_models,
)
from shankh.price_band.validation import walk_forward_folds

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def _set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Config override from JSON file
# ---------------------------------------------------------------------------

def _load_config(override_path: str | None) -> dict:
    cfg = copy.deepcopy(CONFIG)
    if override_path:
        with open(override_path) as f:
            overrides = json.load(f)
        cfg = _deep_merge(cfg, overrides)
        logger.info("Config overrides loaded from %s", override_path)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(cfg: dict) -> dict:
    """
    Execute the full training and evaluation pipeline.

    Parameters
    ----------
    cfg : dict
        Full configuration dict (from config.py, optionally overridden).

    Returns
    -------
    dict
        Evaluation report for programmatic inspection.
    """
    _set_seed(cfg["seed"])

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("STEP 1 — Loading OHLCV data")
    raw_df = load_ohlcv(
        cfg["data"]["data_dir"],
        required_cols=cfg["data"]["required_cols"],
    )
    logger.info(
        "Loaded %d rows | %d tickers | %s → %s",
        len(raw_df),
        raw_df["ticker"].nunique(),
        raw_df["date"].min().date(),
        raw_df["date"].max().date(),
    )

    # ------------------------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------------------------
    logger.info("STEP 2 — Feature engineering")
    feat_df = add_features(raw_df, cfg=cfg["features"])
    feature_cols = get_feature_cols(feat_df)
    logger.info("Feature matrix: %d rows × %d features", len(feat_df), len(feature_cols))

    # ------------------------------------------------------------------
    # 3. Train / test split
    # ------------------------------------------------------------------
    logger.info("STEP 3 — Train/test split (test cutoff: %s)", cfg["data"]["test_cutoff"])
    test_cutoff = cfg["data"]["test_cutoff"]
    pretrain_df = feat_df[feat_df["date"] < test_cutoff].copy().reset_index(drop=True)
    test_df     = feat_df[feat_df["date"] >= test_cutoff].copy().reset_index(drop=True)

    if pretrain_df.empty or test_df.empty:
        raise ValueError(
            f"Train or test set is empty after split at {test_cutoff}. "
            "Check test_cutoff in config."
        )
    logger.info(
        "Pre-test: %d rows | Test: %d rows (%s → %s)",
        len(pretrain_df), len(test_df),
        test_df["date"].min().date(), test_df["date"].max().date(),
    )

    # ------------------------------------------------------------------
    # 4. Walk-forward CV folds
    # ------------------------------------------------------------------
    logger.info("STEP 4 — Generating walk-forward CV folds")
    wf_cfg = cfg["walk_forward"]
    # pretrain_df is already restricted to dates < test_cutoff.
    # walk_forward_folds operates on whatever frame it receives — no
    # cutoff argument needed or accepted.
    folds = walk_forward_folds(
        pretrain_df,
        n_splits       = wf_cfg["n_splits"],
        val_size       = wf_cfg["val_size"],
        min_train_size = wf_cfg["min_train_size"],
        gap            = wf_cfg["gap"],
    )

    # ------------------------------------------------------------------
    # 5. HPO — upper band
    # ------------------------------------------------------------------
    logger.info("STEP 5 — HPO: upper band (q=%.2f)", cfg["model"]["upper_quantile"])
    study_upper, best_upper_params = run_hpo(
        pretrain_df, feature_cols,
        target_col = cfg["targets"]["upper"],
        quantile   = cfg["model"]["upper_quantile"],
        folds      = folds,
        cfg        = cfg,
    )

    # ------------------------------------------------------------------
    # 6. HPO — lower band
    # ------------------------------------------------------------------
    logger.info("STEP 6 — HPO: lower band (q=%.2f)", cfg["model"]["lower_quantile"])
    study_lower, best_lower_params = run_hpo(
        pretrain_df, feature_cols,
        target_col = cfg["targets"]["lower"],
        quantile   = cfg["model"]["lower_quantile"],
        folds      = folds,
        cfg        = cfg,
    )

    # ------------------------------------------------------------------
    # 7. Per-fold CV metrics with best params
    # ------------------------------------------------------------------
    logger.info("STEP 7 — Collecting CV metrics with best params")
    cv_results = collect_cv_metrics(
        pretrain_df, feature_cols,
        best_upper_params, best_lower_params,
        folds, cfg,
    )
    _log_cv_summary(cv_results)

    # ------------------------------------------------------------------
    # 8. Final model training
    # ------------------------------------------------------------------
    logger.info("STEP 8 — Training final models on full pre-test data")
    upper_model, lower_model = train_final_models(
        pretrain_df, feature_cols,
        best_upper_params, best_lower_params,
        cfg,
    )

    # ------------------------------------------------------------------
    # 9. Inference on test set
    # ------------------------------------------------------------------
    logger.info("STEP 9 — Inference on held-out test set")
    preds_df = predict_bands(test_df, upper_model, lower_model, feature_cols)

    pred_upper_ret = preds_df["pred_upper_ret"].values
    pred_lower_ret = preds_df["pred_lower_ret"].values

    # ------------------------------------------------------------------
    # 10. Evaluation
    # ------------------------------------------------------------------
    logger.info("STEP 10 — Full evaluation")
    report = evaluate(test_df, pred_upper_ret, pred_lower_ret, cfg)
    _log_eval_summary(report)

    # ------------------------------------------------------------------
    # 11. Save artifacts
    # ------------------------------------------------------------------
    logger.info("STEP 11 — Saving artifacts")
    save_artifacts(
        upper_model, lower_model,
        feature_cols,
        best_upper_params, best_lower_params,
        study_upper, study_lower,
        cv_results, cfg,
    )
    save_feature_importance(upper_model, lower_model, feature_cols, cfg)
    save_eval_report(report, cfg)
    save_predictions(test_df, pred_upper_ret, pred_lower_ret, cfg)

    logger.info("Pipeline complete.  Artifacts in: %s", cfg["artifacts"]["artifacts_dir"])
    return report


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _log_cv_summary(cv_results: list[dict]) -> None:
    if not cv_results:
        return
    keys = ["pinball_upper", "pinball_lower", "close_coverage_pct", "mean_band_width_pct"]
    header = f"{'Fold':>6}  " + "  ".join(f"{k:>22}" for k in keys)
    logger.info("CV summary:")
    logger.info(header)
    for row in cv_results:
        vals = "  ".join(f"{row.get(k, float('nan')):>22.4f}" for k in keys)
        logger.info(f"{row['fold']:>6}  {vals}")

    mean_vals = {k: np.mean([r.get(k, np.nan) for r in cv_results]) for k in keys}
    logger.info(
        "CV mean  pinball_u=%.5f  pinball_l=%.5f  coverage=%.1f%%  width=%.2f%%",
        mean_vals["pinball_upper"], mean_vals["pinball_lower"],
        mean_vals["close_coverage_pct"], mean_vals["mean_band_width_pct"],
    )


def _log_eval_summary(report: dict) -> None:
    ov = report.get("overall", {})
    bp = report.get("baseline_persistence", {})
    bc = report.get("baseline_constant_vol", {})

    logger.info("=" * 60)
    logger.info("TEST SET EVALUATION")
    logger.info("%-30s  %10s  %10s  %10s", "Metric", "Model", "Persist.", "ConstVol")
    logger.info("-" * 65)
    for metric in [
        "close_coverage_pct", "envelope_coverage_pct",
        "mean_band_width_pct", "mean_winkler_score",
        "mae_high_price", "mae_low_price",
        "pinball_upper", "pinball_lower",
    ]:
        m = ov.get(metric, float("nan"))
        b = bp.get(metric, float("nan"))
        c = bc.get(metric, float("nan"))
        m_str = f"{m:10.4f}" if not np.isnan(m) else "       n/a"
        b_str = f"{b:10.4f}" if not np.isnan(b) else "       n/a"
        c_str = f"{c:10.4f}" if not np.isnan(c) else "       n/a"
        logger.info("%-30s  %s  %s  %s", metric, m_str, b_str, c_str)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Price-band forecasting pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON file with config overrides (merged onto default CONFIG).",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    run(cfg)


if __name__ == "__main__":
    main()
