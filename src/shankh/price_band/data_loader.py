"""
Data loading module for the price-band forecasting pipeline.

Responsibilities
----------------
- Read all per-ticker OHLCV CSV files from a directory.
- Validate schema and data integrity.
- Tag each row with its ticker.
- Return a single chronologically-sorted DataFrame ready for feature engineering.
"""


import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ohlcv(
    data_dir: str | Path,
    required_cols: Optional[list[str]] = None,
    min_rows: int = 30,
) -> pd.DataFrame:
    """
    Load all CSV files in *data_dir*, validate, and concatenate.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing one CSV per ticker.
    required_cols : list[str], optional
        Columns that must be present. Defaults to OHLCV standard.
    min_rows : int
        Skip files with fewer rows than this.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with columns:
        ``date, open, high, low, close, volume, ticker``
        sorted by (ticker, date) and with ``date`` as datetime64.
    """
    if required_cols is None:
        required_cols = ["date", "open", "high", "low", "close", "volume"]

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    frames: list[pd.DataFrame] = []
    for fp in csv_files:
        df = _load_single(fp, required_cols, min_rows)
        if df is not None:
            frames.append(df)

    if not frames:
        raise ValueError("No valid CSV files loaded. Check data_dir and column names.")

    combined = pd.concat(frames, ignore_index=True)
    combined = _validate_and_clean(combined)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    logger.info(
        "Loaded %d rows across %d tickers from %s",
        len(combined),
        combined["ticker"].nunique(),
        data_path,
    )
    return combined


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_single(
    fp: Path,
    required_cols: list[str],
    min_rows: int,
) -> Optional[pd.DataFrame]:
    """Load and do basic validation on a single CSV file."""
    try:
        df = pd.read_csv(fp, low_memory=False)
    except Exception as exc:
        logger.warning("Could not read %s: %s", fp, exc)
        return None

    if df.empty:
        logger.warning("Empty file: %s", fp)
        return None

    missing = set(required_cols) - set(df.columns)
    if missing:
        logger.warning("Skipping %s — missing columns: %s", fp.name, missing)
        return None

    if len(df) < min_rows:
        logger.warning("Skipping %s — only %d rows", fp.name, len(df))
        return None

    # Derive ticker from file stem  (e.g.  "tcs.ns" → "TCS.NS")
    ticker = fp.stem.upper()
    df = df[required_cols].copy()
    df["ticker"] = ticker
    return df


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce column dtypes, remove bad rows, and log data-quality stats.
    """
    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    n_bad_dates = df["date"].isna().sum()
    if n_bad_dates:
        logger.warning("Dropping %d rows with unparseable dates.", n_bad_dates)
    df = df.dropna(subset=["date"])

    # Numeric OHLCV
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)

    # Drop rows where any price is NaN / non-positive
    price_cols = ["open", "high", "low", "close"]
    df = df.dropna(subset=price_cols)
    df = df[(df[price_cols] > 0).all(axis=1)]

    # Enforce OHLC ordering constraints
    df = df[(df["high"] >= df["low"])]
    df = df[(df["high"] >= df["open"]) & (df["high"] >= df["close"])]
    df = df[(df["low"]  <= df["open"]) & (df["low"]  <= df["close"])]

    # Volume: allow NaN but fill with 0
    df["volume"] = df["volume"].fillna(0).clip(lower=0)

    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("Dropped %d rows with invalid OHLCV values.", n_dropped)

    # Remove duplicate (ticker, date) pairs — keep last occurrence
    n_dup = df.duplicated(subset=["ticker", "date"]).sum()
    if n_dup:
        logger.warning("Removing %d duplicate (ticker, date) rows.", n_dup)
        df = df.drop_duplicates(subset=["ticker", "date"], keep="last")

    return df
