"""
Validation module for the stock clustering pipeline.
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def validate_features(factor_df: pd.DataFrame, pivoted_prices: pd.DataFrame) -> bool:
    """
    Validates features for clustering.
    Checks for missing values, NaNs, infinities, and duplicate rows.
    """
    if factor_df.empty or pivoted_prices.empty:
        logger.error("Empty dataframes passed for validation.")
        return False
        
    if factor_df.isnull().any().any():
        logger.error("NaNs found in factor_df.")
        return False
        
    if np.isinf(factor_df.values).any():
        logger.error("Infinities found in factor_df.")
        return False
        
    if factor_df.index.duplicated().any():
        logger.error("Duplicate tickers found in factor_df.")
        return False
        
    logger.info("Features validation passed.")
    return True
