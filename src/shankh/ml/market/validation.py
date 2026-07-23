"""
Validation module for the market regime pipeline.
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def validate_features(regime_df: pd.DataFrame) -> bool:
    """
    Validates features for regime detection.
    Checks for missing values, NaNs, infinities, and duplicate rows.
    """
    if regime_df.empty:
        logger.error("Empty dataframe passed for validation.")
        return False
        
    if regime_df.isnull().any().any():
        logger.error("NaNs found in regime_df.")
        return False
        
    if np.isinf(regime_df.values).any():
        logger.error("Infinities found in regime_df.")
        return False
        
    if regime_df.index.duplicated().any():
        logger.error("Duplicate dates found in regime_df.")
        return False
        
    logger.info("Regime features validation passed.")
    return True
