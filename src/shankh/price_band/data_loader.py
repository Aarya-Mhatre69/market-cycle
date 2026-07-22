# """
# Data loading module for the price-band forecasting pipeline.

# Responsibilities
# ----------------
# - Read all per-ticker OHLCV CSV files from a directory.
# - Validate schema and data integrity.
# - Tag each row with its ticker.
# - Return a single chronologically-sorted DataFrame ready for feature engineering.
# """


# import logging
# from pathlib import Path
# from typing import Optional

# import pandas as pd

# logger = logging.getLogger(__name__)

# # ---------------------------------------------------------------------------
# # Public API
# # ---------------------------------------------------------------------------

# def load_ohlcv(
#     data_dir: str | Path,
#     required_cols: Optional[list[str]] = None,
#     min_rows: int = 30,
# ) -> pd.DataFrame:
#     """
#     Load all CSV files in *data_dir*, validate, and concatenate.

#     Parameters
#     ----------
#     data_dir : str | Path
#         Directory containing one CSV per ticker.
#     required_cols : list[str], optional
#         Columns that must be present. Defaults to OHLCV standard.
#     min_rows : int
#         Skip files with fewer rows than this.

#     Returns
#     -------
#     pd.DataFrame
#         Combined DataFrame with columns:
#         ``date, open, high, low, close, volume, ticker``
#         sorted by (ticker, date) and with ``date`` as datetime64.
#     """
#     if required_cols is None:
#         required_cols = ["date", "open", "high", "low", "close", "volume"]

#     data_path = Path(data_dir)
#     if not data_path.exists():
#         raise FileNotFoundError(f"Data directory not found: {data_path}")

#     csv_files = sorted(data_path.glob("*.csv"))
#     if not csv_files:
#         raise FileNotFoundError(f"No CSV files found in {data_path}")

#     frames: list[pd.DataFrame] = []
#     for fp in csv_files:
#         df = _load_single(fp, required_cols, min_rows)
#         if df is not None:
#             frames.append(df)

#     if not frames:
#         raise ValueError("No valid CSV files loaded. Check data_dir and column names.")

#     combined = pd.concat(frames, ignore_index=True)
#     combined = _validate_and_clean(combined)
#     combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

#     logger.info(
#         "Loaded %d rows across %d tickers from %s",
#         len(combined),
#         combined["ticker"].nunique(),
#         data_path,
#     )
#     return combined


# # ---------------------------------------------------------------------------
# # Internal helpers
# # ---------------------------------------------------------------------------

# def _load_single(
#     fp: Path,
#     required_cols: list[str],
#     min_rows: int,
# ) -> Optional[pd.DataFrame]:
#     """Load and do basic validation on a single CSV file."""
#     try:
#         df = pd.read_csv(fp, low_memory=False)
#     except Exception as exc:
#         logger.warning("Could not read %s: %s", fp, exc)
#         return None

#     if df.empty:
#         logger.warning("Empty file: %s", fp)
#         return None

#     missing = set(required_cols) - set(df.columns)
#     if missing:
#         logger.warning("Skipping %s — missing columns: %s", fp.name, missing)
#         return None

#     if len(df) < min_rows:
#         logger.warning("Skipping %s — only %d rows", fp.name, len(df))
#         return None

#     # Derive ticker from file stem  (e.g.  "tcs.ns" → "TCS.NS")
#     ticker = fp.stem.upper()
#     df = df[required_cols].copy()
#     df["ticker"] = ticker
#     return df


# def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Enforce column dtypes, remove bad rows, and log data-quality stats.
#     """
#     # Parse dates
#     df["date"] = pd.to_datetime(df["date"], errors="coerce")
#     n_bad_dates = df["date"].isna().sum()
#     if n_bad_dates:
#         logger.warning("Dropping %d rows with unparseable dates.", n_bad_dates)
#     df = df.dropna(subset=["date"])

#     # Numeric OHLCV
#     for col in ["open", "high", "low", "close", "volume"]:
#         df[col] = pd.to_numeric(df[col], errors="coerce")

#     n_before = len(df)

#     # Drop rows where any price is NaN / non-positive
#     price_cols = ["open", "high", "low", "close"]
#     df = df.dropna(subset=price_cols)
#     df = df[(df[price_cols] > 0).all(axis=1)]

#     # Enforce OHLC ordering constraints
#     df = df[(df["high"] >= df["low"])]
#     df = df[(df["high"] >= df["open"]) & (df["high"] >= df["close"])]
#     df = df[(df["low"]  <= df["open"]) & (df["low"]  <= df["close"])]

#     # Volume: allow NaN but fill with 0
#     df["volume"] = df["volume"].fillna(0).clip(lower=0)

#     n_dropped = n_before - len(df)
#     if n_dropped:
#         logger.info("Dropped %d rows with invalid OHLCV values.", n_dropped)

#     # Remove duplicate (ticker, date) pairs — keep last occurrence
#     n_dup = df.duplicated(subset=["ticker", "date"]).sum()
#     if n_dup:
#         logger.warning("Removing %d duplicate (ticker, date) rows.", n_dup)
#         df = df.drop_duplicates(subset=["ticker", "date"], keep="last")

#     return df
"""
Data loading module for the price-band forecasting pipeline.
Synchronizes heterogeneous ticker dates to a common, fully aligned overlap range.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ohlcv(
    data_dir: str | Path,
    required_cols: Optional[list[str]] = None,
    min_rows: int = 1000,
    min_tickers: int = 15,
    max_tickers: int = 30,
    min_overlap_days: int = 1000,
) -> pd.DataFrame:
    """
    Load OHLCV files, select active assets, and align them to a common date range.

    Parameters
    ----------
    data_dir : str | Path
        Directory containing one CSV per ticker.
    required_cols : list[str], optional
        Columns that must be present.
    min_rows : int
        Skip raw files with fewer rows than this.
    min_tickers : int
        Minimum number of synchronous tickers to return (must be >= 15).
    max_tickers : int
        Maximum number of synchronous tickers to load.
    min_overlap_days : int
        Minimum overlapping business days required for the synchronous portfolio.

    Returns
    -------
    pd.DataFrame
        Synchronized DataFrame containing aligned tickers sorted by (ticker, date).
    """
    if required_cols is None:
        required_cols = ["date", "open", "high", "low", "close", "volume"]

    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    csv_files = sorted(data_path.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_path}")

    # 1. Load and clean each raw ticker DataFrame individually
    ticker_frames: dict[str, pd.DataFrame] = {}
    for fp in csv_files:
        df = _load_single(fp, required_cols, min_rows)
        if df is not None:
            df_cleaned = _validate_and_clean(df)
            if len(df_cleaned) >= min_rows:
                ticker_frames[df_cleaned["ticker"].iloc[0]] = df_cleaned

    if len(ticker_frames) < min_tickers:
        raise ValueError(
            f"Not enough valid tickers loaded. Have {len(ticker_frames)}, need at least {min_tickers}."
        )

    # 2. Map date spans and identify active (currently trading) tickers
    global_max_date = pd.Timestamp("1970-01-01")
    ticker_stats = []
    for tkr, df in ticker_frames.items():
        t_min = df["date"].min()
        t_max = df["date"].max()
        if t_max > global_max_date:
            global_max_date = t_max
        ticker_stats.append({
            "ticker": tkr,
            "min_date": t_min,
            "max_date": t_max,
        })

    # Filter out historical delisted tickers (max_date must be near the global end-point)
    active_threshold_date = global_max_date - pd.Timedelta(days=35)
    active_tickers = [
        s for s in ticker_stats
        if s["max_date"] >= active_threshold_date
    ]

    if len(active_tickers) < min_tickers:
        logger.warning(
            "Found few active tickers within 35 days of latest date %s. Falling back to all tickers.",
            global_max_date.date()
        )
        active_tickers = ticker_stats

    # Sort active tickers by their start date ascending (longest historical coverage first)
    active_tickers_sorted = sorted(active_tickers, key=lambda x: x["min_date"])

    # 3. Dynamic search for the largest portfolio size M that satisfies overlap requirements
    selected_tickers = []
    common_dates = set()

    for m in range(min(max_tickers, len(active_tickers_sorted)), min_tickers - 1, -1):
        candidate_tkrs = [x["ticker"] for x in active_tickers_sorted[:m]]
        
        # Calculate the mathematical intersection of unique trading dates
        dates_intersection = None
        for tkr in candidate_tkrs:
            tkr_dates = set(ticker_frames[tkr]["date"])
            if dates_intersection is None:
                dates_intersection = tkr_dates
            else:
                dates_intersection = dates_intersection.intersection(tkr_dates)
                
        if dates_intersection and len(dates_intersection) >= min_overlap_days:
            selected_tickers = candidate_tkrs
            common_dates = dates_intersection
            break

    if not selected_tickers:
        raise ValueError(
            f"Could not find a common overlap range of {min_overlap_days} trading dates "
            f"for at least {min_tickers} tickers. Relax constraints in config."
        )

    # 4. Slice DataFrames to the exact synchronous intersection set
    final_frames = []
    for tkr in selected_tickers:
        df_tkr = ticker_frames[tkr]
        df_sliced = df_tkr[df_tkr["date"].isin(common_dates)].copy()
        df_sliced = df_sliced.sort_values("date").reset_index(drop=True)
        final_frames.append(df_sliced)

    combined = pd.concat(final_frames, ignore_index=True)
    combined = combined.sort_values(["ticker", "date"]).reset_index(drop=True)

    logger.info(
        "Aligned %d tickers over a synchronous common date range of %d dates (%s to %s). Total rows: %d.",
        combined["ticker"].nunique(),
        len(common_dates),
        min(common_dates).date(),
        max(common_dates).date(),
        len(combined)
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
        df.columns = df.columns.str.lower()
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

    # Derive ticker from file stem (e.g. "ACC.NS.csv" -> "ACC.NS")
    ticker = fp.stem.upper()
    df = df[required_cols].copy()
    df["ticker"] = ticker
    return df


def _validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce column dtypes, remove bad rows, and log data-quality stats."""
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