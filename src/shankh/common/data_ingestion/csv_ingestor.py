from pathlib import Path

import pandas as pd


REQUIRED_COLS = ["date", "open", "high", "low", "close", "volume"]


def _normalize_ticker_from_filename(path: Path) -> str:
    return path.stem.upper()


def prepare_ticker_frame(csv_path: str | Path) -> pd.DataFrame:
    """
    Load one ticker CSV file.

    Expected input columns:
        date, open, high, low, close, volume

    The ticker is inferred from the file name, e.g.:
        data/tcs.ns.csv -> TCS.NS
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", *REQUIRED_COLS[1:]])

    missing = [col for col in REQUIRED_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path.name}: {missing}")

    ticker = _normalize_ticker_from_filename(path)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=REQUIRED_COLS)
    df["ticker"] = ticker

    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
    df = df.sort_values("date").reset_index(drop=True)

    return df


def load_market_dataset(data_dir: str | Path, pattern: str = "*.csv") -> pd.DataFrame:
    """
    Load all ticker CSV files from a directory and combine them into one dataset.

    Example:
        data/tcs.ns.csv
        data/acc.ns.csv
        data/infy.ns.csv
    """
    base = Path(data_dir)
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {base}")

    files = sorted(base.glob(pattern))
    if not files:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    frames: list[pd.DataFrame] = []
    for file_path in files:
        if file_path.is_file():
            frames.append(prepare_ticker_frame(file_path))

    if not frames:
        return pd.DataFrame(columns=["date", "ticker", "open", "high", "low", "close", "volume"])

    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    return df


__all__ = ["prepare_ticker_frame", "load_market_dataset"]