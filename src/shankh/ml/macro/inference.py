"""
Inference module for the stock clustering pipeline.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import joblib

from src.shankh.ml.macro.config import CONFIG
from src.shankh.ml.macro.features import build_cluster_features
from src.shankh.ml.macro.validation import validate_features
from src.shankh.ml.macro.model import KMeansModel, IsolationForestModel

logger = logging.getLogger(__name__)

def run_inference(df: pd.DataFrame, models_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Run inference using pre-trained clustering and anomaly detection models.
    """
    models_dir = Path(models_dir or CONFIG["artifacts"]["models_dir"])
    
    # Validate input
    if df.empty:
        raise ValueError("Input dataframe is empty.")
        
    # Build features
    pivoted_prices, factor_df = build_cluster_features(df, CONFIG["features"])
    
    if not validate_features(factor_df, pivoted_prices):
        raise ValueError("Feature validation failed during inference.")
        
    # Load scaler
    scaler_path = models_dir / CONFIG["artifacts"]["cluster_scaler"]
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    scaled_factors = scaler.transform(factor_df)
    
    # Load Models
    kmeans_path = models_dir / CONFIG["artifacts"]["kmeans_model"]
    iso_path = models_dir / CONFIG["artifacts"]["isolation_forest"]
    
    kmeans_model = KMeansModel(n_clusters=CONFIG["model"]["n_clusters"])
    kmeans_model.load(kmeans_path)
    
    iso_forest = IsolationForestModel()
    iso_forest.load(iso_path)
    
    # Predict
    kmeans_labels = kmeans_model.predict(scaled_factors)
    kmeans_assignments = {ticker: int(label) for ticker, label in zip(factor_df.index, kmeans_labels)}
    
    anomaly_preds = iso_forest.predict(scaled_factors)
    anomalies = [ticker for ticker, pred in zip(factor_df.index, anomaly_preds) if pred == -1]
    
    return {
        "kmeans_factor_clusters": kmeans_assignments,
        "flagged_forensic_anomalies": anomalies,
    }
