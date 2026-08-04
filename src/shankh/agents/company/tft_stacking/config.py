"""
Configuration module for the Stacking Temporal Fusion Transformer (TFT) pipeline.
"""
from pathlib import Path

# Resolve root paths relative to this file
_SRC_ROOT = Path(__file__).resolve().parents[5]
DATA_DIR = _SRC_ROOT / "data" / "universe"
MODELS_DIR = _SRC_ROOT / "models" / "price_band" / "tft_stacking"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

CONFIG: dict = {
    "seed": 42,
    "data": {
        "data_dir": str(DATA_DIR),
        "test_cutoff": "2023-02-03",
        "required_cols": ["date", "open", "high", "low", "close", "volume"],
        "min_rows": 500,
        "min_tickers": 15,
        "max_tickers": 40,
        "min_overlap_days": 500,
    },
    "features": {
        "rolling_windows": [5, 10, 20, 50],
        "rsi_period": 14,
        "atr_period": 14,
        "bb_period": 20,
        "bb_std": 2.0,
        "vol_windows": [5, 10, 20],
        "ret_lags": [1, 2, 3, 5, 10, 20],
    },
    "tft_dataset": {
        "max_encoder_length": 30,      # 60 trading days historical context (~3 months)
        "max_prediction_length": 1,    # Next-day price-band forecast horizon
        "target": "target_return",     # Next-day log return target
        "group_ids": ["ticker"],
        "static_categoricals": ["ticker", "kmeans_cluster_id"],
        "time_varying_known_categoricals": ["dayofweek", "month", "is_month_end"],
        "time_varying_known_reals": ["time_idx"],
        "time_varying_unknown_reals": [
            "close",
            "log_ret_1",
            "rsi_14",
            "atr_pct",
            "rv_20",
            "garman_klass_20",
            "mkt_volatility",
            "prob_risk_off",
            "lgb_pred_upper",
            "lgb_pred_lower",
            "anomaly_score",
        ],
    },
    "model": {
                "quantiles": [0.16, 0.50, 0.84],  # Added 0.50 for PyTorch Forecasting point metric logging

        "hidden_size": 64,
        "attention_head_size": 4,
        "dropout": 0.2,
        "hidden_continuous_size": 32,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "reduce_on_plateau_patience": 3,
    },
    "trainer": {
        "max_epochs": 5,
        "gradient_clip_val": 0.1,
        "batch_size": 128,
        "num_workers": 2,
        "accelerator": "auto",
        "early_stopping_patience": 5,
    },
    "evaluation": {
        "target_coverage": 0.68,
        "sharpness_quantile": 0.90,
    },
    "artifacts": {
        "models_dir": str(MODELS_DIR),
        "artifacts_dir": str(ARTIFACTS_DIR),
        "tft_checkpoint": "tft_best_checkpoint.ckpt",
        "dataset_parameters": "dataset_params.pkl",
        "eval_report": "eval_report_tft.json",
        "predictions": "tft_predictions.parquet",
    },
}