"""Configuration for the LightGBM price-band forecasting pipeline."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths
# ---------------------------------------------------------------------------
_SRC_ROOT = Path(__file__).resolve().parents[4]
DATA_DIR = _SRC_ROOT / "data" / "universe"
MODELS_DIR = _SRC_ROOT / "models" / "price_band" / "lightgbm"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

# ---------------------------------------------------------------------------
# Master CONFIG dict
# ---------------------------------------------------------------------------
CONFIG: dict = {
    "seed": 42,

    # ---- data -----------------------------------------------------------
    "data": {
        "data_dir": str(DATA_DIR),
        "test_cutoff": "2023-02-03",
        "required_cols": ["date", "open", "high", "low", "close", "volume"],
    },

    # ---- feature engineering -------------------------------------------
    "features": {
        "rolling_windows": [5, 10, 20, 50],
        "rsi_period": 14,
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "vol_windows": [5, 10, 20],
        "ret_lags": [1, 2, 3, 5, 10, 20],
    },

    # ---- targets --------------------------------------------------------
    "targets": {
        "upper": "target_upper",
        "lower": "target_lower",
    },

    # ---- walk-forward validation (Adjusted for 22 Years) -----------------
    "walk_forward": {
        "n_splits": 5,
        "min_train_size": 1250,    # ~5 trading years minimum warmup buffer
        "gap": 1,                  # 1-day purging gap prevents lookahead
        "val_size": 125,           # ~6 months validation per fold
    },

    # ---- Optuna HPO Acceleration ----------------------------------------
    "optuna": {
        "n_trials": 30,
        "direction": "minimize",
        "timeout": None,
        "n_jobs": 4,               # UPDATED: 4 Parallel Workers (4x Speedup!)
        "pruner": "median",
        "n_warmup_steps": 5,
    },

    # ---- model ----------------------------------------------------------
    "model": {
        "name": "lgb",
        "backend": "lightgbm",     # UPDATED: LightGBM for 5x-10x faster execution

        # UPDATED: Calibrated quantiles to fix 85.4% over-coverage -> 68% target
        "upper_quantile": 0.78,    
        "lower_quantile": 0.22,

        # LightGBM Search Space (Fast & Robust)
        "lgb_space": {
            "n_estimators":         [300, 500, 800],
            "learning_rate":        (0.01, 0.08),
            "num_leaves":           [15, 31, 63],
            "max_depth":            [3, 4, 5, 6],
            "min_child_samples":    [20, 50, 100],
            "subsample":            (0.6, 0.9),
            "colsample_bytree":     (0.6, 0.9),
            "reg_alpha":            (0.1, 5.0),
            "reg_lambda":           (0.5, 10.0),
            "min_split_gain":       (0.0, 1.0),  # <--- ADD THIS LINE
            "early_stopping_rounds": [50],
        },
    },

    # ---- evaluation ------------------------------------------------------
    "evaluation": {
        "target_coverage": 0.68,
        "baseline": "persistence",
        "sharpness_quantile": 0.90,
    },

    # ---- artifacts -------------------------------------------------------
    "artifacts": {
        "models_dir":     str(MODELS_DIR),
        "artifacts_dir":  str(ARTIFACTS_DIR),
        "upper_model":    "upper_model",
        "lower_model":    "lower_model",
        "feature_cols":   "feature_cols.pkl",
        "optuna_study":   "optuna_study_{target}.pkl",
        "optuna_best":    "optuna_best.json",
        "eval_report":    "eval_report.json",
        "predictions":    "predictions.parquet",
        "feature_imp":    "feature_importance_{target}.csv",
        "cv_results":     "cv_results.json",
    },
}