"""
Evaluation module for the market regime pipeline.
"""
import logging
from typing import Dict, Any
import pandas as pd

logger = logging.getLogger(__name__)

def evaluate_regime(model: Any, X: Any, regime_df: pd.DataFrame, state_mapping: dict) -> Dict[str, Any]:
    """
    Evaluates the regime model.
    """
    metrics = {}
    
    try:
        metrics["log_likelihood"] = float(model.model.score(X))
    except Exception as e:
        logger.warning(f"Could not compute log likelihood: {e}")
        metrics["log_likelihood"] = None
        
    try:
        transmat = model.model.transmat_
        # Format transition matrix
        trans_dict = {}
        for i in range(transmat.shape[0]):
            state_from = state_mapping.get(i, f"state_{i}")
            trans_dict[state_from] = {}
            for j in range(transmat.shape[1]):
                state_to = state_mapping.get(j, f"state_{j}")
                trans_dict[state_from][state_to] = float(transmat[i, j])
        metrics["transition_matrix"] = trans_dict
        
        # Calculate state duration (1 / (1 - P(i|i)))
        durations = {}
        for i in range(transmat.shape[0]):
            state_label = state_mapping.get(i, f"state_{i}")
            self_trans = transmat[i, i]
            durations[state_label] = float(1 / (1 - self_trans)) if self_trans < 1.0 else float('inf')
        metrics["state_duration_days"] = durations
        
    except Exception as e:
        logger.warning(f"Could not compute transition matrix: {e}")
        
    return metrics
