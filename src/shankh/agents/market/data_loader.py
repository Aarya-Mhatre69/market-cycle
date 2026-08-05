"""
Data loading module for the market regime pipeline.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from shankh.agents.company.data_loader import load_ohlcv as pb_load_ohlcv

logger = logging.getLogger(__name__)


def load_data(
    data_dir: str | Path,
    required_cols: Optional[list[str]] = None,
    min_rows: int = 500,
    min_tickers: int = 15,
    max_tickers: int = 40,
    min_overlap_days: int = 500,
) -> pd.DataFrame:
    """
    Load OHLCV data for market regime analysis.
    Delegates to price_band data loader.
    """
    logger.info("Loading ticker data from universe CSVs...")
    return pb_load_ohlcv(
        data_dir=data_dir,
        required_cols=required_cols,
        min_rows=min_rows,
        min_tickers=min_tickers,
        max_tickers=max_tickers,
        min_overlap_days=min_overlap_days,
    )