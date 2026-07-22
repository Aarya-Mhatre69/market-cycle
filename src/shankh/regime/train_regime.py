"""
src/shankh/analytics/regime_trainer.py

End-to-End Market Regime Analysis Pipeline using Gaussian Hidden Markov Models (HMM).
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class RegimeModelTrainer:
    """
    Constructs market-wide daily macro indicators (volatility, return, breadth,
    and correlation density) and fits a Gaussian Hidden Markov Model (HMM)
    to discover latent market regimes.
    """

    def __init__(self, n_regimes: int = 3, random_state: int = 42) -> None:
        self.n_regimes = n_regimes
        self.random_state = random_state
        self.scaler = StandardScaler()

    # -------------------------------------------------------------------------
    # Feature Engineering
    # -------------------------------------------------------------------------
    def extract_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build daily market-wide regime features from historical OHLCV data.
        """
        logger.info("Computing market-wide regime features...")

        df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

        df["log_return"] = df.groupby("ticker")["close"].transform(
            lambda x: np.log(x / x.shift(1))
        )

        df["sma_20"] = df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(20).mean()
        )

        df["above_20dma"] = (df["close"] > df["sma_20"]).astype(int)
        df["is_positive"] = (df["log_return"] > 0).astype(int)

        pivot_returns = df.pivot(
            index="date",
            columns="ticker",
            values="log_return",
        )

        daily_metrics = []

        for i, dt in enumerate(pivot_returns.index):
            if i < 20:
                continue

            day_data = df[df["date"] == dt]

            if len(day_data) < 5:
                continue

            window = pivot_returns.iloc[i - 20 : i]

            corr = window.corr().values
            triu = np.triu_indices_from(corr, k=1)

            daily_metrics.append(
                {
                    "date": dt,
                    "mkt_return": float(day_data["log_return"].mean()),
                    "mkt_volatility": float(
                        window.mean(axis=1).std() * np.sqrt(252) * 100
                    ),
                    "breadth_pct_above_20dma": float(
                        day_data["above_20dma"].mean() * 100
                    ),
                    "ad_ratio": float(
                        day_data["is_positive"].sum()
                        / max(len(day_data) - day_data["is_positive"].sum(), 1)
                    ),
                    "correlation_density": float(np.nanmean(corr[triu])),
                }
            )

        regime_df = pd.DataFrame(daily_metrics).dropna().set_index("date")

        logger.info(
            "Generated %d daily regime observations.",
            len(regime_df),
        )

        return regime_df

    # -------------------------------------------------------------------------
    # Model Training
    # -------------------------------------------------------------------------
    def train(
        self,
        regime_df: pd.DataFrame,
        output_dir: Path,
    ) -> dict[str, Any]:
        """
        Train Gaussian Hidden Markov Model on market regime features.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        feature_cols = [
            "mkt_volatility",
            "breadth_pct_above_20dma",
            "correlation_density",
        ]

        X = self.scaler.fit_transform(regime_df[feature_cols])

        model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=100,
            random_state=self.random_state,
        )

        model.fit(X)

        regime_df = regime_df.copy()
        regime_df["regime_state"] = model.predict(X)

        state_means = regime_df.groupby("regime_state")[feature_cols].mean()

        # Deterministic State Ranking by Market Volatility Ascending
        sorted_states = state_means.sort_values("mkt_volatility").index.tolist()
        label_hierarchy = ["risk-on", "range-bound", "risk-off"]

        state_mapping: dict[int, str] = {
            int(state_id): label_hierarchy[idx]
            if idx < len(label_hierarchy)
            else f"regime-{idx}"
            for idx, state_id in enumerate(sorted_states)
        }

        regime_df["regime_label"] = regime_df["regime_state"].map(state_mapping)

        # Save Historical Regimes CSV
        regime_df.to_csv(output_dir / "historical_regimes.csv")

        # Save Scaler and Model Artifacts
        joblib.dump(self.scaler, output_dir / "regime_scaler.joblib")
        joblib.dump(model, output_dir / "regime_model.joblib")

        latest_date = regime_df.index[-1]
        latest_date_str = (
            str(latest_date.date())
            if hasattr(latest_date, "date")
            else str(pd.to_datetime(latest_date).date())
        )

        metadata = {
            "model_type": "GaussianHMM",
            "n_regimes": self.n_regimes,
            "state_mapping": state_mapping,
            "state_statistics": state_means.to_dict(),
            "latest_regime": {
                "date": latest_date_str,
                "regime_label": str(regime_df["regime_label"].iloc[-1]),
                "volatility": round(
                    float(regime_df["mkt_volatility"].iloc[-1]),
                    2,
                ),
                "breadth_pct": round(
                    float(regime_df["breadth_pct_above_20dma"].iloc[-1]),
                    2,
                ),
            },
        }

        with open(
            output_dir / "regime_metadata.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            "Saved regime model to %s | Latest regime: %s",
            output_dir,
            metadata["latest_regime"]["regime_label"],
        )

        return metadata

"""
src/shankh/drivers/drive_regime.py
Driver script to run, evaluate, and inspect Market Regime Analysis & State Switches.
"""

import json
import logging
import sys
from pathlib import Path
import pandas as pd

# Path setup to resolve shankh module
PROJECT_ROOT = Path(__file__).resolve().parents[3]
from shankh.price_band.data_loader import load_ohlcv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("drive_regime")


def run_regime_driver():
    data_dir = PROJECT_ROOT / "data" / "universe"
    output_dir = PROJECT_ROOT / "models" / "regime"

    print("=" * 80)
    print(" SHANKH ANALYTICS DRIVER: MARKET REGIME ANALYSIS (HMM)")
    print("=" * 80)
    print(f" Data Directory   : {data_dir}")
    print(f" Artifacts Output : {output_dir}")
    print("-" * 80)

    # 1. Load Real Data from Universe
    logger.info("Loading ticker data from universe CSVs...")
    df = load_ohlcv(
        data_dir=data_dir,
        min_rows=500,
        min_tickers=15,
        max_tickers=40,
        min_overlap_days=500,
    )

    # 2. Extract Macro Regime Features
    trainer = RegimeModelTrainer(n_regimes=3, random_state=42)
    regime_df = trainer.extract_regime_features(df)

    print("\n" + "=" * 80)
    print(" MACRO TIME-SERIES FEATURE HEAD (Sample)")
    print("=" * 80)
    print(regime_df.tail(10).to_string())

    # 3. Train HMM Regime Model
    logger.info("Training Hidden Markov Model for Regime Detection...")
    metadata = trainer.train(regime_df=regime_df, output_dir=output_dir)

    # 4. Display Regime Analysis Report
    print("\n" + "=" * 80)
    print(" CURRENT ACTIVE MARKET REGIME")
    print("=" * 80)
    latest = metadata["latest_regime"]
    print(f"  - As of Date        : {latest['date']}")
    print(f"  - Classified Regime : {latest['regime_label'].upper()}")
    print(f"  - Market Volatility : {latest['volatility']:.2f}% (Annualized)")
    print(f"  - Breadth (> 20-DMA): {latest['breadth_pct']:.1f}% of universe stocks")

    print("\n" + "=" * 80)
    print(" REGIME STATE PROFILES & MEAN STATISTICS")
    print("=" * 80)
    stats_df = pd.DataFrame(metadata["state_statistics"])
    for state_id, label in metadata["state_mapping"].items():
        print(f"\n [State {state_id} -> '{label.upper()}']:")
        if state_id in stats_df.index:
            s_row = stats_df.loc[state_id]
            print(f"   Avg Volatility      : {s_row['mkt_volatility']:.2f}%")
            print(f"   Avg Breadth (>20DMA): {s_row['breadth_pct_above_20dma']:.1f}%")
            print(f"   Correlation Density : {s_row['correlation_density']:.3f}")

    # Load and display recent historical state timeline
    hist_df = pd.read_csv(output_dir / "historical_regimes.csv")
    print("\n" + "=" * 80)
    print(" RECENT HISTORICAL REGIME TIMELINE (Last 15 Trading Days)")
    print("=" * 80)
    recent_history = hist_df[["date", "mkt_volatility", "breadth_pct_above_20dma", "regime_label"]].tail(15)
    print(recent_history.to_string(index=False))

    # Calculate regime distribution over time
    print("\n" + "=" * 80)
    print(" HISTORICAL REGIME FREQUENCY DISTRIBUTION")
    print("=" * 80)
    counts = hist_df["regime_label"].value_counts(normalize=True) * 100
    for r_label, pct in counts.items():
        print(f"  - {r_label.upper():15s} : {pct:5.1f}% of trading history")

    print("\n" + "=" * 80)
    print(" DRIVER RUN COMPLETE: Artifacts saved to 'models/regime/'")
    print("=" * 80)


if __name__ == "__main__":
    run_regime_driver()