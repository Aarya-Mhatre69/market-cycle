"""
Dataset module for preparing time-indexed PyTorch Forecasting TimeSeriesDataSet objects.
"""
import logging
from typing import Tuple
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder

from shankh.agents.company.tft_stacking.meta_features import generate_layer1_meta_features

logger = logging.getLogger(__name__)


def prepare_tft_frame(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Formates OHLCV data, calculates Layer 1 meta-features, creates integer time_idx,
    and structures target returns for TFT consumption.
    """
    # 1. Merge Layer 1 Meta-Features
    df_meta = generate_layer1_meta_features(df, cfg)

    df_work = df_meta.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    df_work["date"] = pd.to_datetime(df_work["date"])

    # 2. Integer continuous time_idx per unique date
    unique_dates = sorted(df_work["date"].unique())
    date_to_idx = {dt: idx for idx, dt in enumerate(unique_dates)}
    df_work["time_idx"] = df_work["date"].map(date_to_idx).astype(int)

    # 3. Categoricals to str
    df_work["ticker"] = df_work["ticker"].astype(str)
    df_work["kmeans_cluster_id"] = df_work["kmeans_cluster_id"].astype(str)
    df_work["dayofweek"] = df_work["date"].dt.dayofweek.astype(str)
    df_work["month"] = df_work["date"].dt.month.astype(str)
    df_work["is_month_end"] = df_work["date"].dt.is_month_end.astype(int).astype(str)

    # 4. Define Target: Next-day close log return
    df_work["target_return"] = df_work.groupby("ticker")["close"].transform(
        lambda s: np.log(s.shift(-1) / (s + 1e-12))
    )

    # 5. Clean numeric columns
    numeric_cols = df_work.select_dtypes(include=[np.number]).columns
    df_work[numeric_cols] = df_work.groupby("ticker")[numeric_cols].transform(
        lambda g: g.bfill().ffill().fillna(0.0)
    )

    # Drop trailing shift(-1) NaNs
    df_clean = df_work.dropna(subset=["target_return"]).reset_index(drop=True)
    logger.info("Prepared TFT Data Frame | Total Rows: %d | Features: %d", len(df_clean), df_clean.shape[1])
    return df_clean


def create_tft_datasets(
    df: pd.DataFrame,
    cfg: dict,
) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet, pd.DataFrame, pd.DataFrame]:
    """
    Constructs training and validation TimeSeriesDataSet objects along with train/test DataFrames.
    """
    tft_cfg = cfg["tft_dataset"]
    test_cutoff = pd.to_datetime(cfg["data"]["test_cutoff"])

    pretrain_df = df[df["date"] < test_cutoff].copy().reset_index(drop=True)
    test_df = df[df["date"] >= test_cutoff].copy().reset_index(drop=True)

    max_time_idx = pretrain_df["time_idx"].max()
    val_size = cfg.get("walk_forward", {}).get("val_size", 125)
    training_cutoff = max_time_idx - val_size

    training_dataset = TimeSeriesDataSet(
        pretrain_df[pretrain_df["time_idx"] <= training_cutoff],
        time_idx="time_idx",
        target=tft_cfg["target"],
        group_ids=tft_cfg["group_ids"],
        min_encoder_length=tft_cfg["max_encoder_length"] // 2,
        max_encoder_length=tft_cfg["max_encoder_length"],
        min_prediction_length=tft_cfg["max_prediction_length"],
        max_prediction_length=tft_cfg["max_prediction_length"],
        static_categoricals=tft_cfg["static_categoricals"],
        time_varying_known_categoricals=tft_cfg["time_varying_known_categoricals"],
        time_varying_known_reals=tft_cfg["time_varying_known_reals"],
        time_varying_unknown_reals=tft_cfg["time_varying_unknown_reals"],
        target_normalizer=GroupNormalizer(
            groups=tft_cfg["group_ids"], transformation="softplus"
        ),
        categorical_encoders={
            "ticker": NaNLabelEncoder(add_nan=True),
            "kmeans_cluster_id": NaNLabelEncoder(add_nan=True),
        },
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )

    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset, pretrain_df, predict=True, stop_randomization=True
    )

    return training_dataset, validation_dataset, pretrain_df, test_df