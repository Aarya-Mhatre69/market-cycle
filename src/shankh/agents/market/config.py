"""
Configuration for the market regime analysis pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths (resolved relative to this file -> project-portable)
# ---------------------------------------------------------------------------
_SRC_ROOT = Path(__file__).resolve().parents[4]   # .../shankh/
DATA_DIR = _SRC_ROOT / "data" / "universe"
MODELS_DIR = _SRC_ROOT / "models" / "regime"
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
        "min_rows": 500,
        "min_tickers": 15,
        "max_tickers": 40,
        "min_overlap_days": 500,
        # columns that must be present in every CSV
        "required_cols": ["date", "open", "high", "low", "close", "volume"],
    },

    # ---- feature engineering -------------------------------------------
    "features": {
        "sma_period": 20,
        "correlation_window": 20,
        "min_day_data": 5,
        "annualization_factor": 252,
    },

    # ---- Optuna HPO -----------------------------------------------------
    "optuna": {
        "n_trials": 40,
        "direction": "maximize",       # maximize log likelihood
        "timeout": None,               # seconds; None = unlimited
        "n_jobs": 1,                   # parallel trials
        "pruner": "median",            # "median" | "hyperband" | None
        "n_warmup_steps": 10,
    },

    # ---- model ----------------------------------------------------------
    "model": {
        "n_regimes": 3,
        
        "hmm": {
            "covariance_type": "full",
            "n_iter": 100,
        }
    },

    # ---- artifacts -------------------------------------------------------
    "artifacts": {
        "models_dir": str(MODELS_DIR),
        "artifacts_dir": str(ARTIFACTS_DIR),
        
        "regime_scaler": "regime_scaler.joblib",
        "regime_model": "regime_model.joblib",
        "historical_regimes": "historical_regimes.csv",
        "regime_metadata": "regime_metadata.json",
        
        "optuna_study": "optuna_study.pkl",
        "optuna_best": "optuna_best.json",
    },
}
