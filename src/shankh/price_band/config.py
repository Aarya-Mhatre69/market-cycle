"""
configuration for the xgboost price-band forecasting pipeline.

"""


from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths  (resolved relative to this file → project-portable)
# ---------------------------------------------------------------------------
_SRC_ROOT = Path(__file__).resolve().parents[3]   # …/shankh/
DATA_DIR   = _SRC_ROOT / "data"
MODELS_DIR = _SRC_ROOT / "models" / "price_band"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

# ---------------------------------------------------------------------------
# Master CONFIG dict
# ---------------------------------------------------------------------------
CONFIG: dict = {
    # ---- reproducibility ------------------------------------------------
    "seed": 42,

    # ---- data -----------------------------------------------------------
    "data": {
        "data_dir": str(DATA_DIR),
        # hard out-of-sample test set starts here
        "test_cutoff": "2024-07-01",
        # columns that must be present in every CSV
        "required_cols": ["date", "open", "high", "low", "close", "volume"],
        # ticker is used ONLY as a grouping key during feature engineering.
        # It is never passed to the model — the model is instrument-agnostic.
    },

    # ---- feature engineering -------------------------------------------
    "features": {
        "rolling_windows": [5, 10, 20, 50],
        "rsi_period": 14,
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        # Parkinson / Garman-Klass volatility estimation windows
        "vol_windows": [5, 10, 20],
        # extra lag returns to compute
        "ret_lags": [1, 2, 3, 5, 10, 20],
    },

    # ---- targets --------------------------------------------------------
    "targets": {
        # Both targets are log-return offsets from today's close.
        # target_upper = log(next_high / close)
        # target_lower = log(next_low  / close)
        "upper": "target_upper",
        "lower": "target_lower",
    },

    # ---- walk-forward validation ----------------------------------------
    "walk_forward": {
        # number of CV folds (expanding window)
        "n_splits": 5,
        # minimum number of *unique trading dates* before the first fold.
        # With 5 tickers × ~250 days/year, 250 unique dates ≈ 1 year of data.
        "min_train_size": 250,
        # unique trading dates to skip between train-end and val-start
        # (prevents look-ahead from multi-day rolling features)
        "gap": 1,
        # number of unique trading dates per validation window
        "val_size": 60,
    },

    # ---- Optuna HPO -----------------------------------------------------
    "optuna": {
        "n_trials": 40,
        "direction": "minimize",       # minimize pinball loss
        "timeout": None,               # seconds; None = unlimited
        "n_jobs": 1,                   # parallel trials
        "pruner": "median",            # "median" | "hyperband" | None
        "n_warmup_steps": 10,
    },

    # ---- model ----------------------------------------------------------
    "model": {
        # "xgboost" | "lightgbm"
        "backend": "lightgbm",

        # quantile levels for the upper / lower bands
        # upper band → high quantile  (e.g. 0.84 ≈ mean + 1 std for normal)
        # lower band → low  quantile
        "upper_quantile": 0.84,
        "lower_quantile": 0.16,

        # XGBoost search space
        "xgb_space": {
            "n_estimators":       [300, 500, 1000, 2000],
            "learning_rate":      (0.005, 0.15),
            "max_depth":          [3, 4, 5, 6, 7],
            "min_child_weight":   [1, 3, 5, 10],
            "subsample":          (0.5, 1.0),
            "colsample_bytree":   (0.5, 1.0),
            "gamma":              (0.0, 2.0),
            "reg_alpha":          (0.0, 5.0),
            "reg_lambda":         (0.5, 10.0),
            "early_stopping_rounds": [50, 100],
        },

        # LightGBM search space
        "lgb_space": {
            "n_estimators":         [300, 500, 1000, 2000],
            "learning_rate":        (0.005, 0.15),
            "num_leaves":           [31, 63, 127, 255],
            "max_depth":            [-1, 4, 6, 8],
            "min_child_samples":    [10, 20, 50, 100],
            "subsample":            (0.5, 1.0),
            "colsample_bytree":     (0.5, 1.0),
            "reg_alpha":            (0.0, 5.0),
            "reg_lambda":           (0.0, 10.0),
            "min_split_gain":       (0.0, 1.0),
            "early_stopping_rounds": [50, 100],
        },
    },

    # ---- evaluation ------------------------------------------------------
    "evaluation": {
        # nominal coverage level we expect from the band
        # e.g. 0.68 → expect ~68 % of next-day closes inside the band
        "target_coverage": 0.68,
        "baseline": "persistence",   # "persistence" | "constant_vol"
        "sharpness_quantile": 0.90,  # width at this percentile is reported
    },

    # ---- artifacts -------------------------------------------------------
    "artifacts": {
        "models_dir":     str(MODELS_DIR),
        "artifacts_dir":  str(ARTIFACTS_DIR),
        # filenames
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
