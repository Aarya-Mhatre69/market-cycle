from pathlib import Path

import pandas as pd

from shankh.common.data_ingestion.csv_ingestor import prepare_ticker_frame
from shankh.common.data_ingestion.csv_ingestor import load_market_dataset

def test_prepare_ticker_frame(tmp_path: Path) -> None:
    csv_path = tmp_path / "tcs.ns.csv"

    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1_000_000, 1_100_000],
        }
    ).to_csv(csv_path, index=False)

    df = prepare_ticker_frame(csv_path)

    assert list(df.columns) == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    assert len(df) == 2
    assert df["ticker"].eq("TCS.NS").all()
    assert df["close"].tolist() == [101.0, 102.0]
    assert df["volume"].tolist() == [1_000_000, 1_100_000]
    assert pd.api.types.is_datetime64_any_dtype(df["date"])




def test_load_market_dataset(tmp_path: Path) -> None:
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1_000_000, 1_100_000],
        }
    ).to_csv(tmp_path / "tcs.ns.csv", index=False)

    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02"],
            "open": [200.0, 201.0],
            "high": [202.0, 203.0],
            "low": [199.0, 200.0],
            "close": [201.0, 202.0],
            "volume": [2_000_000, 2_100_000],
        }
    ).to_csv(tmp_path / "acc.ns.csv", index=False)

    df = load_market_dataset(tmp_path)

    assert len(df) == 4
    assert set(df["ticker"]) == {"ACC.NS", "TCS.NS"}

    assert list(df.columns) == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]

    assert pd.api.types.is_datetime64_any_dtype(df["date"])