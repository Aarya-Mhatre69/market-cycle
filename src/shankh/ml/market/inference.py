"""
Inference module for the market regime pipeline.
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import joblib

from shankh.ml.market.config import CONFIG
from shankh.ml.market.features import build_regime_features
from shankh.ml.market.validation import validate_features
from shankh.ml.market.model import HMMModel

logger = logging.getLogger(__name__)

def run_inference(df: pd.DataFrame, models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run inference using pre-trained HMM regime model.
    """
    models_dir = Path(models_dir or CONFIG["artifacts"]["models_dir"])
    
    # Validate input
    if df.empty:
        raise ValueError("Input dataframe is empty.")
        
    # Build features
    regime_df = build_regime_features(df, CONFIG["features"])
    
    if not validate_features(regime_df):
        raise ValueError("Feature validation failed during inference.")
        
    feature_cols = [
        "mkt_volatility",
        "breadth_pct_above_20dma",
        "correlation_density",
    ]
        
    # Load scaler
    scaler_path = models_dir / CONFIG["artifacts"]["regime_scaler"]
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    X = scaler.transform(regime_df[feature_cols])
    
    # Load Model
    model_path = models_dir / CONFIG["artifacts"]["regime_model"]
    model = HMMModel(n_regimes=CONFIG["model"]["n_regimes"])
    model.load(model_path)
    
    # Predict
    regime_states = model.predict(X)
    regime_df["regime_state"] = regime_states
    
    # Load metadata for state mapping
    meta_path = models_dir / CONFIG["artifacts"]["regime_metadata"]
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)
            state_mapping = metadata.get("state_mapping", {})
            # JSON dict keys are strings, convert to int
            state_mapping = {int(k): v for k, v in state_mapping.items()}
    else:
        logger.warning("Metadata not found, using raw state IDs")
        state_mapping = {i: f"State {i}" for i in range(CONFIG["model"]["n_regimes"])}
        
    regime_df["regime_label"] = regime_df["regime_state"].map(state_mapping)
    
    latest_date = regime_df.index[-1]
    latest_date_str = (
        str(latest_date.date())
        if hasattr(latest_date, "date")
        else str(pd.to_datetime(latest_date).date())
    )
    
    return {
        "date": latest_date_str,
        "regime_label": str(regime_df["regime_label"].iloc[-1]),
        "volatility": round(float(regime_df["mkt_volatility"].iloc[-1]), 2),
        "breadth_pct": round(float(regime_df["breadth_pct_above_20dma"].iloc[-1]), 2),
        "historical_regimes": {
            str(k.date()) if hasattr(k, "date") else str(k): v
            for k, v in regime_df[["regime_state", "regime_label"]].tail(10).to_dict(orient="index").items()
        }
    }
