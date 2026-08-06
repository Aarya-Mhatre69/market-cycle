"""
Configuration for the stock clustering and forensic anomaly detection pipeline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root paths (resolved relative to this file -> project-portable)
# ---------------------------------------------------------------------------
_SRC_ROOT = Path(__file__).resolve().parents[3]   # .../shankh/
DATA_DIR = _SRC_ROOT / "data" / "universe"
MODELS_DIR = _SRC_ROOT / "models" / "clustering"
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
        "min_ticker_rows": 60,
        "rsi_period": 14,
        "atr_period": 14,
        "sma_period": 20,
        "annualization_factor": 252,
        "risk_free_rate": 0.065, # 6.5% RBI repo rate baseline
    },

    # ---- Optuna HPO -----------------------------------------------------
    "optuna": {
        "n_trials": 40,
        "direction": "maximize",       # maximize silhouette score
        "timeout": None,               # seconds; None = unlimited
        "n_jobs": 1,                   # parallel trials
        "pruner": "median",            # "median" | "hyperband" | None
        "n_warmup_steps": 10,
    },

    # ---- model ----------------------------------------------------------
    "model": {
        "n_clusters": 4, # Used in KMeans and AgglomerativeClustering
        
        "agglomerative": {
            "metric": "precomputed",
            "linkage": "average",
        },
        "kmeans": {
            "n_init": 10,
        },
        "isolation_forest": {
            "contamination": 0.10,
        }
    },

    # ---- artifacts -------------------------------------------------------
    "artifacts": {
        "models_dir": str(MODELS_DIR),
        "artifacts_dir": str(ARTIFACTS_DIR),
        
        "cluster_scaler": "cluster_scaler.joblib",
        "kmeans_model": "kmeans_cluster_model.joblib",
        "isolation_forest": "isolation_forest_model.joblib",
        "cluster_results": "cluster_results.json",
        
        "optuna_study": "optuna_study.pkl",
        "optuna_best": "optuna_best.json",
    },
}
