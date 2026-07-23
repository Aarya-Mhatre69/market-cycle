"""
Evaluation module for the stock clustering pipeline.
"""
import logging
from typing import Dict, Any

from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

logger = logging.getLogger(__name__)

def evaluate_clustering(scaled_factors: Any, labels: Any) -> Dict[str, float]:
    """
    Evaluates clustering performance.
    """
    try:
        sil_score = float(silhouette_score(scaled_factors, labels))
        db_score = float(davies_bouldin_score(scaled_factors, labels))
        ch_score = float(calinski_harabasz_score(scaled_factors, labels))
        
        return {
            "silhouette_score": round(sil_score, 4),
            "davies_bouldin_index": round(db_score, 4),
            "calinski_harabasz_index": round(ch_score, 4),
        }
    except ValueError as e:
        logger.warning(f"Could not compute clustering metrics: {e}")
        return {
            "silhouette_score": 0.0,
            "davies_bouldin_index": 0.0,
            "calinski_harabasz_index": 0.0,
        }
