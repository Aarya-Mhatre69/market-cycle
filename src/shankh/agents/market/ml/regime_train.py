"""
Pipeline entrypoint for Market Regime Analysis Training.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from shankh.agents.market.ml.regime_config import CONFIG
from shankh.agents.market.ml.regime_data_loader import load_data
from shankh.agents.market.ml.regime_features import build_regime_features
from shankh.agents.market.ml.regime_model import HMMModel
from shankh.agents.market.ml.regime_evaluation import evaluate_regime


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_regime")

FEATURE_COLS = [
    "mkt_return_20d",
    "mkt_volatility",
    "parkinson_volatility",
    "composite_breadth",
    "ad_index",
    "volume_breadth_ratio",
    "correlation_density",
]


def train_pipeline() -> Dict[str, Any]:
    """
    End-to-End training flow for Market Regime Analysis (HMM).
    """
    data_dir = Path(CONFIG["data"]["data_dir"])
    output_dir = Path(CONFIG["artifacts"]["models_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" SHANKH ANALYTICS DRIVER: MARKET REGIME ANALYSIS (HMM)")
    print("=" * 80)
    print(f" Data Directory   : {data_dir}")
    print(f" Artifacts Output : {output_dir}")
    print("-" * 80)

    # 1. Load Data
    df = load_data(
        data_dir=data_dir,
        min_rows=CONFIG["data"]["min_rows"],
        min_tickers=CONFIG["data"]["min_tickers"],
        max_tickers=CONFIG["data"]["max_tickers"],
        min_overlap_days=CONFIG["data"]["min_overlap_days"],
        required_cols=CONFIG["data"]["required_cols"]
    )

    # 2. Engineer Features
    regime_df = build_regime_features(df, CONFIG["features"])

    print("\n" + "=" * 80)
    print(" MACRO TIME-SERIES FEATURE HEAD (Sample)")
    print("=" * 80)
    print(regime_df.tail(10).to_string())

    logger.info("Training Hidden Markov Model for Regime Detection...")
    
    # 3. Scale Features & Fit HMM Model
    scaler = StandardScaler()
    X = scaler.fit_transform(regime_df[FEATURE_COLS])

    model = HMMModel(
        n_regimes=CONFIG["model"]["n_regimes"],
        random_state=CONFIG["seed"],
        **CONFIG["model"]["hmm"]
    )
    model.fit(X)

    regime_df = regime_df.copy()
    regime_df["regime_state"] = model.predict(X)

    # Compute mean statistics per state
    state_means = regime_df.groupby("regime_state")[FEATURE_COLS].mean()

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
    
    # 4. Evaluate Regime
    metrics = evaluate_regime(model, X, regime_df, state_mapping)

    # 5. Save Artifacts
    regime_df.to_csv(output_dir / CONFIG["artifacts"]["historical_regimes"])
    joblib.dump(scaler, output_dir / CONFIG["artifacts"]["regime_scaler"])
    model.save(output_dir / CONFIG["artifacts"]["regime_model"])

    latest_date = regime_df.index[-1]
    latest_date_str = (
        str(latest_date.date())
        if hasattr(latest_date, "date")
        else str(pd.pd.to_datetime(latest_date).date()) if hasattr(pd, "pd") else str(pd.to_datetime(latest_date).date())
    )

    metadata = {
        "model_type": "GaussianHMM",
        "n_regimes": CONFIG["model"]["n_regimes"],
        "state_mapping": state_mapping,
        "state_statistics": state_means.to_dict(),
        "evaluation_metrics": metrics,
        "latest_regime": {
            "date": latest_date_str,
            "regime_label": str(regime_df["regime_label"].iloc[-1]),
            "volatility": round(
                float(regime_df["mkt_volatility"].iloc[-1]),
                2,
            ),
            "breadth_pct": round(
                float(regime_df["composite_breadth"].iloc[-1]),
                2,
            ),
        },
    }

    with open(output_dir / CONFIG["artifacts"]["regime_metadata"], "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Saved regime model to %s | Latest regime: %s",
        output_dir,
        metadata["latest_regime"]["regime_label"],
    )

    # 6. Console Summary Outputs
    print("\n" + "=" * 80)
    print(" CURRENT ACTIVE MARKET REGIME")
    print("=" * 80)
    latest = metadata["latest_regime"]
    print(f"  - As of Date        : {latest['date']}")
    print(f"  - Classified Regime : {latest['regime_label'].upper()}")
    print(f"  - Market Volatility : {latest['volatility']:.2f}% (Annualized)")
    print(f"  - Composite Breadth : {latest['breadth_pct']:.1f}% of universe stocks")

    print("\n" + "=" * 80)
    print(" REGIME STATE PROFILES & MEAN STATISTICS")
    print("=" * 80)
    stats_df = pd.DataFrame(metadata["state_statistics"])
    for state_id, label in metadata["state_mapping"].items():
        print(f"\n [State {state_id} -> '{label.upper()}']:")
        if state_id in stats_df.index:
            s_row = stats_df.loc[state_id]
            print(f"   Avg Volatility      : {s_row['mkt_volatility']:.2f}%")
            print(f"   Composite Breadth   : {s_row['composite_breadth']:.1f}%")
            print(f"   Correlation Density : {s_row['correlation_density']:.3f}")

    hist_df = pd.read_csv(output_dir / CONFIG["artifacts"]["historical_regimes"])
    print("\n" + "=" * 80)
    print(" RECENT HISTORICAL REGIME TIMELINE (Last 15 Trading Days)")
    print("=" * 80)
    recent_history = hist_df[["date", "mkt_volatility", "composite_breadth", "regime_label"]].tail(15)
    print(recent_history.to_string(index=False))

    print("\n" + "=" * 80)
    print(" HISTORICAL REGIME FREQUENCY DISTRIBUTION")
    print("=" * 80)
    counts = hist_df["regime_label"].value_counts(normalize=True) * 100
    for r_label, pct in counts.items():
        print(f"  - {r_label.upper():15s} : {pct:5.1f}% of trading history")

    print("\n" + "=" * 80)
    print(f" DRIVER RUN COMPLETE: Artifacts saved to '{output_dir}'")
    print("=" * 80)
    
    return metadata


if __name__ == "__main__":
    train_pipeline()